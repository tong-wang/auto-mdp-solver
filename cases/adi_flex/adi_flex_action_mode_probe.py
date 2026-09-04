"""Acceptance tests for the action modes whose decode the IR cannot declare.

Schema v1 refuses a per-component ``transform`` on a multi-decision action mode
(upstream, F30's limitation on its second occurrence), so ``order_protection``,
``target_ip`` and ``target_mip`` all ship ``transform: ""`` and their decodes
live in ``adi_flex_gym`` alone. Nothing gates them: the differential drives the
mdp layer through ``mdp.decisions``, never through a gym mode, so conformance,
laws and the differential are all blind to a wrong decode.

These are the checks that stand in for the gate that does not exist. Each one
replays a policy whose value is already known to the float through the NEW mode
and demands the SAME number — not a tolerance, the same float:

1. ``order_protection`` @ het_exp4, sigma_1 held at PL(sigma)'s own protection
   level, must reproduce the PL(sigma) arm exactly. At T_dl = 2 the cascade IS
   paper sec 4.2's rule, so any difference is a decode defect.
2. ``order_protection`` @ het_exp7, sigma pinned to 0. lambda_0 = 0 there, so
   no reserve can ever be useful and the mode must degenerate to forced maximal
   fill -- which sec 3 proves optimal for a homogeneous base -- against the
   exact DP.
3. ``target_mip`` @ homog_L0_T2 and homog_L0_Tdl1, replaying the solver's own
   y* table. The DP/AP benchmark policies are ALREADY order-up-to policies
   (``adi_flex_benchmark_rule`` computes ``clip(y - u, 0, order_max)`` from a
   table whose columns are ``(i, vhat, s, S)``), so the optimal policy is
   natively expressible in this mode and the replay must reproduce the DP.
4. ``target_ip`` @ homog_L0_T2, replaying the same table re-referenced to the
   inventory position. IP and u differ by ``sum(adv)``, which the decode
   subtracts and the replay adds back, so this checks the OTHER reference
   without needing a second solver.

Every arm is played on the same CRN seed block as the benchmarks, so the
comparison is against the recorded number rather than a re-measurement.

    python adi_flex_action_mode_probe.py                 # all, 2048 seeds
    python adi_flex_action_mode_probe.py --n-seeds 8192  # the full protocol
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from adi_flex_benchmark_common import ObsView
from adi_flex_benchmark_rule import U_MIN, protection_level
from adi_flex_gym import PROTECT_MODES, TARGET_MODES, AdiFlexEnv
from adi_flex_scenarios import SCENARIOS

HERE = Path(__file__).resolve().parent


def _load_table(path: Path) -> tuple[np.ndarray, int, int]:
    z = np.load(path)
    table = z["y"] if "y" in z else z[z.files[0]]
    u_min = int(z["u_min"]) if "u_min" in z.files else U_MIN
    d_max = int(z["d_max"]) if "d_max" in z.files else (table.shape[-1] - 1)
    return table, u_min, d_max


def _lookup(table, u_min, d_max, u, vhat, period) -> int:
    n_u = table.shape[1] if table.ndim == 3 else table.shape[0]
    ui = int(np.clip(u - u_min, 0, n_u - 1))
    vi = int(np.clip(vhat, 0, d_max))
    return int(table[period, ui, vi] if table.ndim == 3 else table[ui, vi])


def run_arm(scenario_name: str, action_mode: str, n_seeds: int,
            table_path: Path, sigma: int = 0) -> float:
    """Play ONE policy through ONE action mode; return the mean COST.

    The observation is always `vec` regardless of the mode under test: both
    decodes read the STATE internally, and this replay is not an agent, so the
    arm's own observation pairing (target_mip with vec_mip, etc.) is irrelevant
    here. `ObsView` decodes the `vec` layout only.
    """
    sc = SCENARIOS[scenario_name]
    env = AdiFlexEnv(scenario=sc, action_mode=action_mode,
                     observation_mode="vec")
    table, u_min, d_max = _load_table(table_path)

    rewards = np.zeros(n_seeds)
    for ep_seed in range(n_seeds):
        obs, _ = env.reset(seed=ep_seed)
        terminated = False
        while not terminated:
            # gated on `two_phase`, NOT alloc_enabled: order_protection has a
            # live allocation and no interior, so it renders no tail block
            v = ObsView(obs, env.pipe_slots, sc.N, sc.T_dl,
                        alloc_enabled=env.two_phase)
            if action_mode in PROTECT_MODES:
                # one action per period: an order head followed by n_sigma
                # reserve levels. The head is a QUANTITY for order_protection
                # and a TARGET (with its index-0 no-order sentinel) for the
                # combined target_*_protection modes — the two decodes compose,
                # which is the whole claim those modes make.
                y = _lookup(table, u_min, d_max, v.u, v.vhat, v.period)
                order = int(np.clip(y - v.u, 0, sc.order_max))
                if action_mode in TARGET_MODES:
                    ref = v.u + (sum(v.adv) if "mip" not in action_mode else 0)
                    head = (0 if order == 0
                            else int(np.clip(ref + order, 0, sc.order_max)) + 1)
                else:
                    head = order
                act = [head] + [int(sigma)] * env.n_sigma
            elif v.phase == 0:
                y = _lookup(table, u_min, d_max, v.u, v.vhat, v.period)
                if action_mode in ("target_mip", "target_ip"):
                    # re-express the SAME order as a representable target.
                    # index 0 is the no-order sentinel; otherwise the target
                    # that reproduces this order is clip(ref + order, 0, cap),
                    # which is exact even where the solver's own y is negative
                    # (that only happens where order_max binds, and the clipped
                    # target yields the same capped order).
                    order = int(np.clip(y - v.u, 0, sc.order_max))
                    if order == 0:
                        act = 0
                    else:
                        ref = v.u + (sum(v.adv) if action_mode == "target_ip" else 0)
                        act = int(np.clip(ref + order, 0, sc.order_max)) + 1
                else:
                    act = int(np.clip(y - v.u, 0, sc.order_max))  # quantity
            else:
                # §4.2: full fill of the due-next class, then protect sigma
                act = (min(v.surplus, v.outstanding) if len(env._alloc_buf) == 0
                       else int(np.clip(v.surplus - sigma, 0, v.outstanding)))
            obs, _, terminated, _, _ = env.step(act)
        rewards[ep_seed] = env.total_reward
    env.close()
    return float(-rewards.mean())


# Each case is PAIRED: the same policy played through the incumbent mode and
# through the mode under test, on the same CRN seeds. The bar is EQUALITY, not
# proximity to a recorded number — a recorded number is measured at a different
# seed count and is not the same quantity.
CASES = [
    ("order_protection == PL(sigma) @ T_dl=2", "het_exp4",
     "results/het_exp4/ap/het_exp4_policy.npz", "plsigma", "order_protection",
     "the cascade IS §4.2's rule, so any gap is a decode defect"),
    ("order_protection == forced fill (lambda_0=0)", "het_exp7",
     "results/het_exp7/ap/het_exp7_policy.npz", 0, "order_protection",
     "sigma_max derives to 0 there: no reserve can ever be useful"),
    ("target_mip == quantity, DP policy", "homog_L0_T2",
     "results/homog_L0_T2/dp/homog_L0_T2_policy.npz", 0, "target_mip",
     "the DP's own table is already an order-up-to policy on u"),
    ("target_mip == quantity, DP policy", "homog_L0_Tdl1",
     "results/homog_L0_Tdl1/dp/homog_L0_Tdl1_policy.npz", 0, "target_mip",
     "Case-1 board: V~ is empty, so u is the bare scalar"),
    ("target_ip == quantity, DP policy", "homog_L0_T2",
     "results/homog_L0_T2/dp/homog_L0_T2_policy.npz", 0, "target_ip",
     "the OTHER reference: IP = u + sum(adv), checked without a second solver"),
    ("target_mip == quantity, AP policy", "het_exp4",
     "results/het_exp4/ap/het_exp4_policy.npz", "plsigma", "target_mip",
     "the target decode on the het board, allocation held at PL(sigma)"),
    # --- the combined modes: BOTH decodes at once, one action per period -----
    ("target_ip_protection == PL(sigma)", "het_exp4",
     "results/het_exp4/ap/het_exp4_policy.npz", "plsigma", "target_ip_protection",
     "order-up-to on IP AND protection levels, composed; must still be §4.2"),
    ("target_mip_protection == PL(sigma)", "het_exp4",
     "results/het_exp4/ap/het_exp4_policy.npz", "plsigma", "target_mip_protection",
     "same, on the mip reference — the narrowest observation the domain admits"),
    ("target_ip_protection == forced fill", "het_exp7",
     "results/het_exp7/ap/het_exp7_policy.npz", 0, "target_ip_protection",
     "lambda_0 = 0: sigma vanishes, so only the target decode is under test"),
    ("target_mip_protection == DP policy", "homog_L0_T2",
     "results/homog_L0_T2/dp/homog_L0_T2_policy.npz", 0, "target_mip_protection",
     "homogeneous: the allocation is inert, so this isolates the order decode"),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-seeds", type=int, default=2048)
    a = p.parse_args()

    print(f"action-mode acceptance — PAIRED, {a.n_seeds} CRN seeds")
    print("same policy, two encodings; the bar is equality to the float")
    print("-" * 96)
    print(f"  {'case':<44}{'seq_mask':>13}{'under test':>13}{'delta':>12}  verdict")
    fails = 0
    for name, scen, tbl, sig, mode, why in CASES:
        sc = SCENARIOS[scen]
        sigma = protection_level(sc, sig) if isinstance(sig, str) else sig
        ref = run_arm(scen, "seq_mask", a.n_seeds, HERE / tbl, sigma)
        got = run_arm(scen, mode,       a.n_seeds, HERE / tbl, sigma)
        d = got - ref
        ok = d == 0.0
        fails += (not ok)
        print(f"  {name:<44}{ref:13.6f}{got:13.6f}{d:12.6f}  "
              f"{'EXACT' if ok else '*** DIFFERS ***'}")
        print(f"      {why}")
    print("-" * 96)
    print(f"  {len(CASES) - fails}/{len(CASES)} exact")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()

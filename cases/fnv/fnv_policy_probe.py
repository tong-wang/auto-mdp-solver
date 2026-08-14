"""Policy-interpretation probe for FNV — the anchored readback (spec §14.1/§14.2).

FNV ships an exact DP reference (fnv_benchmark_dp.py), so the anchored form is
required: read the trained net back into the domain's policy vocabulary and say,
as a number, whether it implements the predicted structure.

Predicted structural class: **modified base-stock in the reduced coordinate**.
The DP optimum orders up to

    S_n = mu + I + b_n      (additive)      S_n = exp(mu + I + b_n)  (multiplicative)

so in canonical coordinates the post-decision level `x + a` is FLAT at S_n over
every state that still acts, and follows the `y = x` no-act diagonal above it.
The probe measures four things per period:

  flatness    spread of (x + a) over the acting region; 0 = exact base-stock
  b_hat       the recovered offset, = mean(x + a) - mu - I over acting states
  agreement   fraction of probed states where the net's action matches the DP's
  sensitivity how far the action moves when one observation feature is swept

Then (§14.2) the fitted rule is itself scored as a policy, on the same CRN block
as every other arm, and the trio reference / fitted rule / raw net is reported
with paired Δs. The fitted rule can BEAT the net it was read from — fitting a
constant offset deletes the net's raggedness — in which case a few numbers per
period are the shippable artifact.

Verdict branches are pre-decided (§14.2):
  1. clean recovery and |fitted - net| <= 1% of the bar -> claim the structure;
  2. |fitted - net| > 1% of the bar -> the net is doing something else, diagnose;
  3. no structure recovered -> suspect the action interface, not the theory.

Example usage:
    python fnv_policy_probe.py --model-path results/simple/PPO_.../simple_ppo.zip \\
        -s simple --dp-solutions results/simple/benchmark/dp/simple.txt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from fnv_benchmark_dp import solve_offsets
from fnv_benchmark_dp_eval import make_dp_act_batch
from fnv_ppo_eval import (
    DEFAULT_N_SEEDS,
    PROTOCOL_SEED_BASE,
    load_obs_rms,
    normalize_obs,
    resolve_cells,
    resolve_vecnorm_path,
    rollout_seeds,
)
from fnv_grids import GRIDS
from fnv_scenarios import FnvScenario, SCENARIOS

# States whose order quantity exceeds this are "acting" — below it the policy is
# effectively on the no-act diagonal and carries no order-up-to information.
ACTING_TOL = 1e-3


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Probe a trained FNV policy (§14).")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--vecnorm-path", type=str, default=None)
    p.add_argument("-s", "--scenario_name", type=str, default="simple",
                   choices=list(SCENARIOS) + list(GRIDS))
    p.add_argument("--cell", type=str, default=None,
                   help="For a grid, the cell id to probe (default: first cell)")
    p.add_argument("--dp-solutions", type=str, default=None,
                   help="Offsets TSV; solved on the fly when omitted")
    p.add_argument("--n-states", type=int, default=201,
                   help="Points on the swept inventory axis")
    p.add_argument("--information", type=float, default=0.0,
                   help="Held value of the information coordinate during the sweep")
    p.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS,
                   help="CRN block for scoring the fitted rule (§14.2)")
    p.add_argument("--skip-scoring", action="store_true",
                   help="Surface + fit only; do not score the fitted rule")
    p.add_argument("--outdir", type=str, default=None,
                   help="Run dir to anchor outputs to (default: the model's own "
                        "run dir; probe/ and interpret/ are created inside it)")
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def build_obs(scenario: FnvScenario, period: int, inventory, information) -> np.ndarray:
    """Observation rows in mode 'vec': [stdev, T, lamb, period, inventory, information]."""
    inventory   = np.atleast_1d(np.asarray(inventory, dtype=np.float32))
    information = np.broadcast_to(
        np.asarray(information, dtype=np.float32), inventory.shape)
    n = len(inventory)
    return np.column_stack([
        np.full(n, scenario.stdev), np.full(n, scenario.T), np.full(n, scenario.lamb),
        np.full(n, float(period)), inventory, information,
    ]).astype(np.float32)


def sweep_period(
    act_batch, scenario: FnvScenario, period: int,
    inventory_grid: np.ndarray, information: float,
) -> pd.DataFrame:
    """Raw action surface at one period — dumped so fits never re-run the sweep."""
    obs     = build_obs(scenario, period, inventory_grid, information)
    actions = np.asarray(act_batch(obs)).reshape(-1)
    return pd.DataFrame({
        "period":      period,
        "inventory":   inventory_grid,
        "information": information,
        "action":      actions,
        "post_level":  inventory_grid + actions,     # canonical coordinate
    })


def fit_offset(surface: pd.DataFrame, scenario: FnvScenario) -> dict:
    """Recover the base-stock offset and state how well the form holds.

    flatness is the spread of the post-decision level over the acting region —
    a number, not an impression. An exact base-stock policy scores 0.
    """
    acting = surface[surface["action"] > ACTING_TOL]
    if len(acting) < 2:
        return {"period": int(surface["period"].iloc[0]), "n_acting": len(acting),
                "b_hat": float("nan"), "flatness": float("nan"),
                "S_hat": float("nan")}

    if scenario.mmfe_mode == "additive":
        level = acting["post_level"].to_numpy()
        b_hat = float(level.mean() - scenario.mu - acting["information"].iloc[0])
    else:
        # multiplicative: S = exp(mu + I + b), so read the offset in logs
        level = np.log(np.maximum(acting["post_level"].to_numpy(), 1e-12))
        b_hat = float(level.mean() - scenario.mu - acting["information"].iloc[0])

    return {
        "period":   int(surface["period"].iloc[0]),
        "n_acting": int(len(acting)),
        "b_hat":    b_hat,
        # spread in the coordinate the structure is flat in
        "flatness": float(level.max() - level.min()),
        "S_hat":    float(acting["post_level"].mean()),
    }


def recover_level(
    act_batch, scenario: FnvScenario, period: int,
    inventory_grid: np.ndarray, information: float,
) -> dict:
    """Recover the order-up-to level S_n at ONE information value.

    Returned in the coordinate the paper's structure is linear in: raw `S` for
    a-MMFE, `log S` for m-MMFE. `spread` is the flatness of that level over the
    acting region — 0 means an exact base-stock policy at this I.
    """
    surface = sweep_period(act_batch, scenario, period, inventory_grid, information)
    acting = surface[surface["action"] > ACTING_TOL]
    if len(acting) < 2:
        return {"information": information, "level": float("nan"),
                "spread": float("nan"), "n_acting": len(acting)}
    lvl = acting["post_level"].to_numpy()
    if scenario.mmfe_mode != "additive":
        lvl = np.log(np.maximum(lvl, 1e-12))
    return {"information": information, "level": float(lvl.mean()),
            "spread": float(lvl.max() - lvl.min()), "n_acting": int(len(acting))}


def reachable_states(act_batch, scenario: FnvScenario, n_episodes: int = 512) -> pd.DataFrame:
    """Roll the policy out and record the (period, inventory, information) it
    actually visits.

    A trained net is only meaningful on its own state distribution. Sweeping
    inventory over a wide grid at period 1 — where inventory is identically 0,
    because nothing has been ordered yet — measures the network off-manifold
    and reports raggedness that the policy never exhibits in play.
    """
    from fnv_mdp import advance, init_state

    rows = []
    for ep in range(n_episodes):
        state, _ = init_state(scenario, ep)
        while not state.terminated:
            rows.append({"period": state.period, "inventory": state.inventory,
                         "information": state.information})
            obs = build_obs(scenario, state.period, [state.inventory], state.information)
            q = float(np.asarray(act_batch(obs)).reshape(-1)[0])
            state, _ = advance(scenario, state, max(0.0, q))
    return pd.DataFrame(rows)


def linear_response(
    act_batch, scenario: FnvScenario, dp_offsets: np.ndarray,
    reach: pd.DataFrame, n_information: int = 21, n_x: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test Proposition 2 in its sharpest form: does the order respond to new
    information ONE-FOR-ONE?

        S_n(I) = mu + I + b_n     and     a = S_n(I) - x
        =>   d a / d I = 1   at fixed x,   intercept = mu + b_n - x

    Working on the action rather than on a fitted "level" avoids needing the
    base-stock region to be flat first: unit response is exactly the claim that
    b_n does not depend on the forecast evolution. Slope < 1 means the policy
    under-reacts to information; > 1, over-reacts.

    Probed at inventory quantiles the policy actually reaches, and over the
    range of I actually realized at that period. **Period 1 is excluded: I_1 is
    identically 0** (the first increment is revealed only after the period-1
    order), so there is no information variation to regress against — the
    claim is vacuous there and only S_1 vs mu + b_1 is checkable.
    """
    add = scenario.mmfe_mode == "additive"
    rows, fits = [], []

    for period in range(1, scenario.N + 1):
        sub = reach[reach["period"] == period]
        if len(sub) == 0:
            continue
        i_sd = float(sub["information"].std(ddof=1)) if len(sub) > 1 else 0.0

        if i_sd < 1e-9:
            # degenerate: information is deterministic at this period (n = 1)
            x0 = float(sub["inventory"].median())
            obs = build_obs(scenario, period, [x0], float(sub["information"].median()))
            a = float(np.asarray(act_batch(obs)).reshape(-1)[0])
            level = (x0 + a) if add else math.log(max(x0 + a, 1e-12))
            fits.append({"period": period, "x": x0, "slope": float("nan"),
                         "intercept": float("nan"), "r2": float("nan"),
                         "b_hat": level - scenario.mu,
                         "b_dp": float(dp_offsets[period - 1]),
                         "b_err": level - scenario.mu - float(dp_offsets[period - 1]),
                         "note": "I deterministic; slope untestable"})
            continue

        lo, hi = sub["information"].quantile([0.05, 0.95])
        info_grid = np.linspace(float(lo), float(hi), n_information)
        # Deduplicate: because I_1 = 0, the period-1 action is deterministic and
        # the reachable inventory at later periods can collapse to a single
        # point. Probing "quantiles" of a degenerate distribution would just
        # repeat one row and fake independent evidence.
        xs = np.unique(np.round(
            sub["inventory"].quantile(np.linspace(0.1, 0.9, n_x)).to_numpy(), 9))

        for x0 in xs:
            obs = build_obs(scenario, period, np.full(n_information, x0), 0.0)
            obs[:, 5] = info_grid
            acts = np.asarray(act_batch(obs)).reshape(-1)
            for I, a in zip(info_grid, acts):
                rows.append({"period": period, "inventory": float(x0),
                             "information": float(I), "action": float(a),
                             "post_level": float(x0 + a)})

            y = (x0 + acts) if add else np.log(np.maximum(x0 + acts, 1e-12))
            # regress the achieved level on I: slope 1 <=> unit response
            slope, intercept = np.polyfit(info_grid, y, 1)
            resid = y - (slope * info_grid + intercept)
            ss = float(((y - y.mean()) ** 2).sum())
            fits.append({
                "period": period, "x": float(x0),
                "slope": float(slope),
                "intercept": float(intercept),
                "r2": 1.0 - float((resid ** 2).sum()) / ss if ss > 0 else float("nan"),
                "b_hat": float(intercept) - scenario.mu,
                "b_dp": float(dp_offsets[period - 1]),
                "b_err": float(intercept) - scenario.mu - float(dp_offsets[period - 1]),
                "note": "",
            })

    return pd.DataFrame(rows), pd.DataFrame(fits)


def linear_structure(
    act_batch, scenario: FnvScenario, dp_offsets: np.ndarray,
    n_information: int = 21, n_states: int = 201, span: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test Proposition 2: is the recovered base-stock level LINEAR in I,
    with unit slope and intercept mu + b_n?

        a-MMFE   S_n(I)     = mu + I + b_n
        m-MMFE   log S_n(I) = mu + I + b_n

    Unit slope is the sharp claim: the base-stock level tracks the information
    state one-for-one, which is equivalent to the safety stock b_n being
    independent of the forecast evolution. A slope below 1 means the policy
    under-reacts to new information; above 1, it over-reacts.

    Returns (per-I levels, per-period regression summary).
    """
    sigma = max(float(scenario.stdev), 1e-9)
    info_grid = np.linspace(-span * sigma, span * sigma, n_information)

    rows, fits = [], []
    for period in range(1, scenario.N + 1):
        # the acting region moves up with I, so the inventory sweep must cover
        # the highest level any probed I can induce
        hi = scenario.mu + info_grid.max() + 4.0 * sigma
        if scenario.mmfe_mode != "additive":
            hi = float(np.exp(scenario.mu + info_grid.max() + 4.0 * sigma))
        inventory_grid = np.linspace(0.0, hi, n_states)

        levels = [recover_level(act_batch, scenario, period, inventory_grid, I)
                  for I in info_grid]
        for r in levels:
            r["period"] = period
        rows.extend(levels)

        df = pd.DataFrame(levels).dropna(subset=["level"])
        if len(df) < 3:
            fits.append({"period": period, "slope": float("nan"),
                         "intercept": float("nan"), "r2": float("nan"),
                         "b_hat": float("nan"), "b_dp": float(dp_offsets[period - 1]),
                         "b_err": float("nan"), "max_flatness": float("nan")})
            continue

        x, y = df["information"].to_numpy(), df["level"].to_numpy()
        slope, intercept = np.polyfit(x, y, 1)
        resid = y - (slope * x + intercept)
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
        # Proposition 2: intercept = mu + b_n, so the recovered safety stock is
        # whatever the intercept leaves over after the mean demand term
        b_hat = float(intercept) - scenario.mu
        fits.append({
            "period": period,
            "slope": float(slope),            # predicted 1.0
            "intercept": float(intercept),    # predicted mu + b_n
            "r2": r2,                         # predicted 1.0 (exact linearity)
            "b_hat": b_hat,
            "b_dp": float(dp_offsets[period - 1]),
            "b_err": b_hat - float(dp_offsets[period - 1]),
            "max_flatness": float(df["spread"].max()),
        })

    return pd.DataFrame(rows), pd.DataFrame(fits)


def action_agreement(
    surface: pd.DataFrame, scenario: FnvScenario, offsets: np.ndarray, tol: float,
) -> float:
    """Fraction of probed states where the net's action matches the DP's."""
    obs = build_obs(scenario, int(surface["period"].iloc[0]),
                    surface["inventory"].to_numpy(), surface["information"].to_numpy())
    dp  = make_dp_act_batch(offsets, scenario.mu,
                            scenario.mmfe_mode == "additive")(obs).reshape(-1)
    return float(np.mean(np.abs(dp - surface["action"].to_numpy()) <= tol))


def feature_sensitivity(
    act_batch, scenario: FnvScenario, period: int, inventory: float, information: float,
) -> pd.DataFrame:
    """Hold the state fixed, sweep one feature, report how far the action moves.

    The mechanism check (§14.1): it is independent of any score verdict. For a
    base-stock policy the response to `information` is the informative one —
    the optimum shifts its target one-for-one with I, so d(action)/dI = 1.
    """
    rows = []
    base = build_obs(scenario, period, [inventory], information)
    base_action = float(np.asarray(act_batch(base)).reshape(-1)[0])

    sweeps = {
        "information": np.linspace(information - 2 * scenario.stdev,
                                   information + 2 * scenario.stdev, 21),
        "stdev": np.linspace(max(1e-3, scenario.stdev * 0.5), scenario.stdev * 1.5, 21),
        "lamb":  np.linspace(max(0.0, scenario.lamb * 0.5), scenario.lamb * 1.5, 21),
        "T":     np.linspace(max(0.05, scenario.T * 0.5), min(0.95, scenario.T * 1.5), 21),
    }
    col = {f: i for i, f in enumerate(
        ("stdev", "T", "lamb", "period", "inventory", "information"))}

    for feature, values in sweeps.items():
        obs = np.repeat(base, len(values), axis=0)
        obs[:, col[feature]] = values
        actions = np.asarray(act_batch(obs)).reshape(-1)
        slope = float(np.polyfit(values, actions, 1)[0]) if np.ptp(values) > 0 else 0.0
        rows.append({
            "period": period, "feature": feature,
            "range": float(np.ptp(values)),
            "action_span": float(actions.max() - actions.min()),
            "slope": slope,
            "base_action": base_action,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Scoring the fitted rule (§14.2)
# ---------------------------------------------------------------------------

def score_arms(
    act_batch, scenario: FnvScenario, fitted: np.ndarray, dp_offsets: np.ndarray,
    n_seeds: int, seed_base: int,
) -> pd.DataFrame:
    """Score reference / fitted rule / raw net on one shared CRN block."""
    seeds    = range(seed_base, seed_base + n_seeds)
    additive = scenario.mmfe_mode == "additive"

    arms = {
        "reference_dp": make_dp_act_batch(dp_offsets, scenario.mu, additive),
        "fitted_rule":  make_dp_act_batch(fitted, scenario.mu, additive),
        "raw_net":      act_batch,
    }
    per_seed, rows = {}, []
    for name, fn in arms.items():
        profits, _ = rollout_seeds(fn, scenario, seeds, batch_size=512)
        per_seed[name] = profits
        rows.append({"arm": name, "profit_mean": float(profits.mean()),
                     "profit_se": float(profits.std(ddof=1) / np.sqrt(n_seeds))})

    out = pd.DataFrame(rows)
    # paired deltas: the shared seed block is common random numbers, so instance
    # luck cancels and the paired SE is far smaller than the per-arm SE
    for a, b in (("fitted_rule", "raw_net"), ("raw_net", "reference_dp"),
                 ("fitted_rule", "reference_dp")):
        d = per_seed[a] - per_seed[b]
        out.loc[len(out)] = {
            "arm": f"Δ {a} - {b}",
            "profit_mean": float(d.mean()),
            "profit_se": float(d.std(ddof=1) / np.sqrt(n_seeds)),
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    model_path   = Path(args.model_path)
    vecnorm_path = resolve_vecnorm_path(model_path, args.vecnorm_path)
    # §8.4 write-direction boundary: this script plays both roles, so it keeps
    # them in separate trees. probe/ holds raw instrument output (regenerable
    # measurements); interpret/ holds the conclusions drawn from it.
    run_dir      = Path(args.outdir) if args.outdir else model_path.parent
    probe_dir    = run_dir / "probe"
    interpret_dir = run_dir / "interpret"
    probe_dir.mkdir(parents=True, exist_ok=True)
    interpret_dir.mkdir(parents=True, exist_ok=True)

    cells = resolve_cells(args.scenario_name)
    if args.cell:
        cells = [c for c in cells if c[0] == args.cell] or cells[:1]
    cell_id, scenario = cells[0]

    model = PPO.load(str(model_path), device="cpu")
    obs_rms, clip_obs = load_obs_rms(vecnorm_path)

    def act_batch(obs: np.ndarray) -> np.ndarray:
        action, _ = model.predict(
            normalize_obs(np.asarray(obs, dtype=np.float32), obs_rms, clip_obs),
            deterministic=True)
        return np.asarray(action).reshape(-1, 1)

    # the reference the readback is anchored to
    if args.dp_solutions:
        sol = pd.read_csv(args.dp_solutions, delimiter="\t")
        from fnv_benchmark_dp_eval import lookup_offsets
        dp_offsets = lookup_offsets(sol, scenario)
        if dp_offsets is None:
            raise SystemExit(f"no DP row for cell {cell_id}")
    else:
        dp_offsets = np.array(solve_offsets(scenario))

    print(f"model     {model_path}")
    print(f"scenario  {args.scenario_name}  cell {cell_id}")
    print(f"vecnorm   {vecnorm_path}  (obs_rms={'loaded' if obs_rms is not None else 'NONE'})")
    print(f"DP offsets b = {np.round(dp_offsets, 5).tolist()}")
    print(f"probe     {probe_dir}")
    print(f"interpret {interpret_dir}\n")

    # --- action surface + structural fit, per period ----------------------
    hi = scenario.mu + 4.0 * scenario.stdev
    if scenario.mmfe_mode == "multiplicative":
        hi = float(np.exp(scenario.mu + 4.0 * scenario.stdev))
    inventory_grid = np.linspace(0.0, hi, args.n_states)

    surfaces, fits, sens = [], [], []
    action_tol = 0.02 * max(hi, 1e-9)
    for period in range(1, scenario.N + 1):
        surface = sweep_period(act_batch, scenario, period,
                               inventory_grid, args.information)
        fit = fit_offset(surface, scenario)
        fit["b_dp"] = float(dp_offsets[period - 1])
        fit["b_err"] = fit["b_hat"] - fit["b_dp"]
        fit["agreement"] = action_agreement(surface, scenario, dp_offsets, action_tol)
        surfaces.append(surface)
        fits.append(fit)
        sens.append(feature_sensitivity(
            act_batch, scenario, period, inventory=0.0, information=args.information))

    # --- Proposition 2: unit response to information, on reachable states ---
    reach = reachable_states(act_batch, scenario)
    levels_df, linfit_df = linear_response(act_batch, scenario, dp_offsets, reach)

    surface_df = pd.concat(surfaces, ignore_index=True)
    fit_df     = pd.DataFrame(fits)
    sens_df    = pd.concat(sens, ignore_index=True)

    # raw measurements -> probe/ ; fitted form + agreement + verdict -> interpret/
    surface_df.to_csv(probe_dir / "action_surface.tsv", sep="\t", index=False)
    sens_df.to_csv(probe_dir / "feature_sensitivity.tsv", sep="\t", index=False)
    levels_df.to_csv(probe_dir / "level_vs_information.tsv", sep="\t", index=False)
    fit_df.to_csv(interpret_dir / "structural_fit.tsv", sep="\t", index=False)
    linfit_df.to_csv(interpret_dir / "linear_structure.tsv", sep="\t", index=False)

    print("structural fit at fixed I (flatness 0 = exact base-stock):")
    print(fit_df[["period", "n_acting", "b_hat", "b_dp", "b_err",
                  "flatness", "agreement"]].to_string(index=False))

    coord = "S" if scenario.mmfe_mode == "additive" else "log S"
    print(f"\n=== PROPOSITION 2: is {coord}_n = mu + I + b_n ? ===")
    print("    on REACHABLE states only; predicted slope = 1.000, r2 = 1.000,")
    print("    intercept - mu = b_n.  Period 1 excluded: I_1 is identically 0.")
    print(linfit_df[["period", "x", "slope", "intercept", "r2",
                     "b_hat", "b_dp", "b_err"]]
          .to_string(index=False, float_format=lambda x: f"{x: .4f}"))
    for n in sorted(set(linfit_df.loc[linfit_df["note"] != "", "period"])):
        print(f"    note (period {n}): "
              f"{linfit_df.loc[linfit_df['period'] == n, 'note'].iloc[0]}")

    good = linfit_df.dropna(subset=["slope"])
    if len(good):
        worst_slope = float((good["slope"] - 1.0).abs().max())
        mean_slope  = float(good["slope"].mean())
        min_r2      = float(good["r2"].min())
        worst_b     = float(good["b_err"].abs().max())
        print(f"\n  slope: mean {mean_slope:.4f}, max |slope - 1| = {worst_slope:.4f}"
              f"   min r2 = {min_r2:.5f}   max |b_hat - b_dp| = {worst_b:.4f}")
        if min_r2 >= 0.99 and worst_slope <= 0.10:
            print("  => LINEAR STRUCTURE RECOVERED: the level is affine in I with "
                  "unit slope — the net implements the paper's state-dependent "
                  "base-stock, and its safety stock is independent of I.")
        elif min_r2 >= 0.99:
            print(f"  => AFFINE BUT NOT UNIT SLOPE (mean {mean_slope:.3f}): the net "
                  f"{'under' if mean_slope < 1 else 'over'}-reacts to information, "
                  f"so its implied b_n still depends on I — the linear FORM is "
                  f"recovered, the unit COEFFICIENT is not.")
        else:
            print("  => NOT LINEAR: suspect the action interface before the "
                  "theory (§14.2 branch 3).")

    print("\nfeature sensitivity (slope vs the swept feature):")
    print(sens_df[["period", "feature", "action_span", "slope"]].to_string(index=False))

    # --- score the fitted rule (§14.2) ------------------------------------
    if args.skip_scoring:
        print("\n(scoring skipped)")
        return

    # Full-horizon assertion (§14.2): a rule fitted on some periods acts
    # arbitrarily in the rest and returns a plausible-looking garbage score.
    if fit_df["b_hat"].isna().any():
        missing = fit_df.loc[fit_df["b_hat"].isna(), "period"].tolist()
        raise SystemExit(
            f"fit does not cover periods {missing} (no acting states probed): "
            f"the rule cannot be scored over the full horizon. Widen the "
            f"inventory sweep or lower --information.")
    assert len(fit_df) == scenario.N, "fit must cover every period of the horizon."

    fitted = fit_df["b_hat"].to_numpy()
    print(f"\nscoring on {args.n_seeds} CRN seeds from {PROTOCOL_SEED_BASE} …")
    scores = score_arms(act_batch, scenario, fitted, dp_offsets,
                        args.n_seeds, PROTOCOL_SEED_BASE)
    # §8.4: a fitted rule scored against a benchmark compares ACROSS policies,
    # so it is not about this one model — it belongs in the domain-level
    # insights/ tree, not in the run dir.
    insights = (Path(__file__).resolve().parent / "results" / "insights"
                / "fitted-rule-vs-benchmark")
    insights.mkdir(parents=True, exist_ok=True)
    scores_path = insights / f"scores_{args.scenario_name}_{cell_id}.tsv"
    scores.to_csv(scores_path, sep="\t", index=False)
    print(scores.to_string(index=False))
    print(f"\nscored arms → {scores_path}")

    bar   = float(scores.loc[scores["arm"] == "reference_dp", "profit_mean"].iloc[0])
    net   = float(scores.loc[scores["arm"] == "raw_net", "profit_mean"].iloc[0])
    rule  = float(scores.loc[scores["arm"] == "fitted_rule", "profit_mean"].iloc[0])
    gap   = rule - net
    # absolute gap beside the percentage (§14.2): bars differ across cells, so a
    # percentage alone does not compare
    print(f"\n|fitted - net| = {abs(gap):.6f} absolute "
          f"({100 * abs(gap) / abs(bar):.3f}% of the {bar:.6f} bar)")
    if fit_df["flatness"].max() > 0.1 * max(scenario.stdev, 1e-9) * 10:
        print("VERDICT (3): no clean structure recovered — suspect the action "
              "interface (encoding, bounds) before the theory.")
    elif abs(gap) <= 0.01 * abs(bar):
        print("VERDICT (1): the net implements the predicted base-stock structure.")
    else:
        print("VERDICT (2): the net is doing something else — diagnose before "
              "claiming the structural form.")
    if rule > net:
        print(f"NOTE: the fitted rule BEATS the net by {rule - net:+.6f} — "
              f"{scenario.N} numbers are a shippable artifact (§14.2).")

    print(f"\nmeasurements → {probe_dir}\nconclusions  → {interpret_dir}")
    print(json.dumps({"cell": cell_id, "b_hat": fitted.tolist(),
                      "b_dp": dp_offsets.tolist()}, indent=2))


if __name__ == "__main__":
    main()

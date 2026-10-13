"""Deployable cnv policy (spec §12): a trained artifact behind ``act(obs)``, no SB3/Gym knowledge needed.

``CnvPolicy`` bundles the three parts of a trained model's I/O contract — the SB3 model (CPU), the
VecNormalize observation statistics saved beside it (applied as ``clip((obs − mean) / sqrt(var + eps))``;
the net was trained on normalised inputs, so they are mandatory), and the action transform. The shipped
artifacts all train under ``order_discrete``, whose transform is the IR's identity (the action IS the
stocking level, floored at 0 — ``cnv_gym.CnvEnv.decode_action``); other action modes are refused.

Observation contract — one per-period vector per call, float32, in this order (``cnv_gym.render_obs``
renders it from the scenario, the state and the last period's outcome; a caller with its own records
supplies the same numbers):

| observation_mode | features |
|---|---|
| ``timing_belief_ctx`` | time_to_go, critical_ratio = (r − c) / r, prior_rate, prior_mean, posterior shape = prior_shape + cumulative sales, posterior rate = prior_rate + cumulative sales duration (periods, fractional when stocked out) |
| ``full_belief_ctx`` | time_to_go, critical_ratio, prior_rate, prior_mean, prior_shape + cumulative demand, prior_rate + periods elapsed |
| ``timing_belief`` / ``full_belief`` | the ``_ctx`` vectors without prior_rate, prior_mean |
| ``timing_belief_log`` / ``full_belief_log`` | time_to_go, critical_ratio, log(posterior mean), 1 / sqrt(posterior shape) |
| ``event_raw`` | time_to_go, critical_ratio, prior_rate, prior_mean, last period's sales, last period's stock-out flag (0/1) |
| ``timing_raw`` | the same four, last period's sales, last period's stock-out time (fraction of the period; 1.0 = no stock-out) |
| ``full_raw`` | the same four, last period's demand |

time_to_go = periods left including the current one (N at the first decision); the first decision of an
episode sees zeros for the last-period outcome. The raw modes were trained on a frame stack of the last
``frame_stack`` vectors (zero-padded, newest last): ``CnvPolicy`` keeps that stack itself — pass one
vector per period and call ``reset()`` at the start of every episode.

``observation_mode``, ``action_mode`` and ``scenario`` must match training: the constructor checks them
against the run's ``{scenario}_ppo_args.txt``.

    python cnv_policy.py                                  # smoke test: the shipped paper510 generalists
    python cnv_policy.py --model-path <zip> -o <mode> -s <scenario> --seeds-file <protocol .seeds.tsv>
"""

from __future__ import annotations

import argparse
import glob
import pickle
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# the shipped artifacts (README §Results, paper510): one generalist per observation type the operator can record
SHIPPED = {
    "timing": ("results/tuning/cnv_paper510_ppo_timing_belief_ctx_all/trial_0062/paper510/PPO_*/paper510_ppo.zip",
               "timing_belief_ctx", "results/tuning/cnv_paper510_ppo_timing_belief_ctx_all/winner_protocol_paper510.seeds.tsv"),
    "full": ("results/tuning/cnv_paper510_ppo_full_belief_ctx_all/trial_0062/paper510/PPO_*/paper510_ppo.zip",
             "full_belief_ctx", "results/tuning/cnv_paper510_ppo_full_belief_ctx_all/winner_protocol_paper510.seeds.tsv"),
    "event": ("results/tuning/cnv_paper510_ppo_event_raw_union/winner/PPO_*/paper510_ppo.zip",
              "event_raw", "results/tuning/cnv_paper510_ppo_event_raw_union/winner_protocol_paper510.seeds.tsv"),
}


def _run_args(model_path: Path) -> dict:
    for d in (model_path.parent, model_path.parent.parent):   # a checkpoint's run dir is one level up
        for f in d.glob("*_ppo_args.txt"):
            return dict(ln.split(": ", 1) for ln in f.read_text().splitlines() if ": " in ln)
    raise FileNotFoundError(f"no *_ppo_args.txt beside {model_path}")


class CnvPolicy:
    def __init__(self, model_path, vecnorm_path=None, action_mode: str = "order_discrete",
                 observation_mode: str = "timing_belief_ctx", scenario: str = "paper510",
                 deterministic: bool = True):
        from stable_baselines3 import PPO
        self.model_path = Path(model_path).resolve()
        run = _run_args(self.model_path)
        for key, val in (("observation_mode", observation_mode), ("action_mode", action_mode), ("scenario_name", scenario)):
            if run.get(key) != val:
                raise ValueError(f"{key}={val!r} but the run trained under {run.get(key)!r} ({self.model_path})")
        if action_mode != "order_discrete":
            raise ValueError("CnvPolicy ships order_discrete artifacts only")
        self.observation_mode, self.action_mode, self.scenario = observation_mode, action_mode, scenario
        self.frame_stack = int(run.get("frame_stack", "1") or 1)
        self.deterministic = deterministic
        try:
            self.model = PPO.load(str(self.model_path), device="cpu")
        except Exception:   # SB3 version skew: the schedules are training-only
            self.model = PPO.load(str(self.model_path), device="cpu",
                                  custom_objects={"lr_schedule": lambda _: 0.0, "learning_rate": 0.0,
                                                  "clip_range": lambda _: 0.0})
        if vecnorm_path is None:
            ckpt = self.model_path.parent.name == "checkpoints"
            vecnorm_path = (self.model_path.with_name(self.model_path.name.replace("ppo_cnv_", "ppo_cnv_vecnormalize_").replace(".zip", ".pkl"))
                            if ckpt else self.model_path.parent / "vecnormalize.pkl")
        self.norm = None
        if Path(vecnorm_path).exists():
            with open(vecnorm_path, "rb") as f:
                vn = pickle.load(f)
            if vn.norm_obs:
                self.norm = (vn.obs_rms.mean.astype(np.float64), vn.obs_rms.var.astype(np.float64),
                             float(vn.epsilon), float(vn.clip_obs))
        self.reset()

    def reset(self) -> None:
        """Start an episode: clears the raw modes' frame stack."""
        self._frames = None

    def act(self, obs) -> int:
        """The stocking level for this period's observation vector (deterministic by default)."""
        x = np.asarray(obs, dtype=np.float32)
        if self.frame_stack > 1:
            if self._frames is None:
                self._frames = deque([np.zeros_like(x)] * self.frame_stack, maxlen=self.frame_stack)
            self._frames.append(x)
            x = np.stack(self._frames)
        if self.norm is not None:
            mean, var, eps, clip = self.norm
            x = np.clip((x - mean) / np.sqrt(var + eps), -clip, clip).astype(np.float32)
        action, _ = self.model.predict(x[None, ...], deterministic=self.deterministic)
        return max(0, int(np.asarray(action).reshape(-1)[0]))


class PremiumRule:
    """The readback's fitted rule (spec §14.2; INTERPRET.md): order = y^M + premium[period], y^M the myopic
    newsvendor level of the observation TYPE. The paper510 rules distilled from the generalists:
    timing +1 at the first decision, event +2, full none (``SHIPPED_RULES``).

    Observation contract: the same per-period vector ``CnvPolicy`` takes —
    timing / full: a ``*_belief`` or ``*_belief_ctx`` vector (y^M is read off its last two entries, the
    posterior shape and rate); event: an ``event_raw`` vector (the rule carries the benchmark's grid
    posterior itself, from each period's sales and stock-out flag and its own last order). Needs the
    episode's scenario (horizon, critical ratio, prior, arrival buffer). Call ``reset()`` every episode."""

    def __init__(self, obs_type: str, premium_by_period, scenario):
        assert obs_type in ("timing", "full", "event"), obs_type
        self.obs_type, self.sc = obs_type, scenario
        self.premium = [int(premium_by_period[p]) for p in range(int(scenario.horizon))]
        self.observation_mode = {"timing": "timing_belief_ctx", "full": "full_belief_ctx", "event": "event_raw"}[obs_type]
        self.reset()

    def reset(self) -> None:
        self._ev, self._last = None, 0

    def act(self, obs) -> int:
        from cnv_bayes import gamma_poisson_quantile
        x = np.asarray(obs, dtype=np.float64)
        period = int(round(self.sc.horizon - x[0]))
        if self.obs_type in ("timing", "full"):
            y_m = gamma_poisson_quantile(float(x[-2]), float(x[-1]), float(x[1]))
        else:
            from types import SimpleNamespace
            from cnv_benchmark_myopic import MyopicEvent
            shim = SimpleNamespace(_scenario_ep=self.sc, _state=SimpleNamespace(period=period),
                                   _info={"sales": int(x[4]), "stockout_event": int(x[5]), "order": self._last})
            if self._ev is None:
                self._ev = MyopicEvent(shim)
            y_m = self._ev(shim)
        self._last = max(0, y_m + self.premium[period])
        return self._last


# the paper510 fitted rules (INTERPRET.md, #E118): each generalist's rounded mean premium by period
SHIPPED_RULES = {"timing": [1, 0, 0, 0], "full": [0, 0, 0, 0], "event": [2, 0, 0, 0]}


def replay(policy: CnvPolicy, source, seed: int) -> float:
    """One episode on the raw cnv_mdp loop (no gym): the profit the policy earns."""
    import cnv_mdp as mdp
    from cnv_gym import render_obs
    sc = source(seed) if callable(source) else source
    st, info = mdp.init_state(sc, seed)
    policy.reset()
    profit = 0.0
    while not st.terminated:
        order = policy.act(render_obs(policy.observation_mode, sc, st, info))
        st, info = mdp.advance(sc, st, order)
        profit += float(info["profit"]["total"])
    return profit


def _smoke(model_path: Path, mode: str, scenario: str, seeds_file: str | None, n_targets: int, n_seeds: int) -> bool:
    import pandas as pd
    from cnv_grids import GRIDS
    from cnv_scenarios import SCENARIOS
    pol = CnvPolicy(model_path, observation_mode=mode, scenario=scenario)
    targets = list(GRIDS[scenario])[:n_targets] if scenario in GRIDS else [(scenario, SCENARIOS[scenario])]
    ref = pd.read_csv(seeds_file, sep="\t").set_index(["target", "seed"])["profit"] if seeds_file else None
    ok = True
    for cid, src in targets:
        for seed in range(n_seeds):                          # the protocol block starts at seed 0
            got = replay(pol, src, seed)
            want = float(ref.loc[(cid, seed)]) if ref is not None else None
            same = want is None or abs(got - want) < 1e-6
            ok &= same
            print(f"  {cid} seed {seed}: profit {got:10.4f}" + ("" if want is None else
                  f"   protocol record {want:10.4f}   {'MATCH' if same else 'DIFF'}"))
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description="Smoke-test CnvPolicy on the raw cnv_mdp loop.")
    p.add_argument("--model-path", default=None)
    p.add_argument("-o", "--observation_mode", default=None)
    p.add_argument("-s", "--scenario", default="paper510")
    p.add_argument("--seeds-file", default=None, help="the artifact's protocol per-seed sidecar, for the match")
    p.add_argument("--n-targets", type=int, default=3)
    p.add_argument("--n-seeds", type=int, default=3)
    a = p.parse_args()
    jobs = ([(Path(a.model_path), a.observation_mode, a.scenario, a.seeds_file)] if a.model_path else
            [(Path(sorted(glob.glob(str(HERE / m)))[0]), mode, "paper510", str(HERE / s)) for m, mode, s in SHIPPED.values()])
    ok = True
    for mp, mode, sc, sf in jobs:
        print(f"[smoke] {mode} — {mp.parent.name[:80]}")
        ok &= _smoke(mp, mode, sc, sf, a.n_targets, a.n_seeds)
    if not a.model_path:                                     # the shipped fitted rules, against their protocol record
        import pandas as pd
        from cnv_grids import GRIDS
        record = {"timing": "tb_ctx", "full": "fb_ctx", "event": "e_raw"}
        for kind, prem in SHIPPED_RULES.items():
            f = HERE / f"results/paper510/benchmark/benchmark_fitted_{record[kind]}_period_eval_paper510.seeds.tsv"
            ref = pd.read_csv(f, sep="\t").set_index(["target", "seed"])["profit"] if f.exists() else None
            print(f"[smoke] PremiumRule {kind} {prem}" + ("" if ref is not None else "  (no protocol record yet)"))
            for cid, src in list(GRIDS["paper510"])[:a.n_targets]:
                rule = PremiumRule(kind, prem, src(0))
                for seed in range(a.n_seeds):
                    got = replay(rule, src, seed)
                    want = float(ref.loc[(cid, seed)]) if ref is not None else None
                    same = want is None or abs(got - want) < 1e-6
                    ok &= same
                    print(f"  {cid} seed {seed}: profit {got:10.4f}" + ("" if want is None else
                          f"   protocol record {want:10.4f}   {'MATCH' if same else 'DIFF'}"))
    print("[smoke] all episodes reproduce the protocol record" if ok else "[smoke] MISMATCH — see above")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

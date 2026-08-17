"""Train PPO on the serial multi-echelon chain (spec §8).

Training levels (spec §8.6)
---------------------------
``--level L1`` (default) is the mandatory run: the §8.6 derivation table applied
to this domain, every derived knob logged with its rationale in
``{scenario}_ppo_args.txt``.

``--level L0`` is the control: SB3's library defaults, ``gamma = beta``, and no
VecNormalize. It is the ruler for how much the derivation actually bought, and
is reporting-only — never a gate.

Derivations for this domain, each one line:

* ``gamma = 0.95`` — the *default*, taken from the IR's
  ``objective.discount_factor`` (beta), because gamma = beta is the faithful
  starting point (spec §8.2). gamma is nonetheless a **PPO training knob**,
  distinct from beta: beta fixes how the eval and every benchmark score
  ``sum beta^t cost_t``, while gamma only shapes how far ahead the algorithm
  assigns credit while learning. Moving it is legitimate — spec §8.2 allows
  ``gamma < beta`` as a *logged escalation move*, and ``gamma > beta`` never.
  Rewards are never pre-discounted in the env either way.
* ``n_envs = 4``, ``n_steps = 2048`` — rollout 8192 transitions >= max(2048,
  10 episodes x T=50 = 500). Structural, never tuned.
* ``gae_lambda = 0.95`` — with beta = 0.95 the effective horizon is ~20 periods
  against T = 50, so credit does not need to travel the full episode.
* VecNormalize on — the IR's ``obs_normalization`` decision. The observation box
  is a validity envelope sized to the reachable range, so raw magnitudes are far
  larger than typical play; normalization is what makes them learnable.
* ``ent_coef = 0.005`` — a small floor on a continuous Gaussian policy. Raw
  action 0 means "ship nothing", a live low-value action, so there is no dead
  zone to escape; the floor only slows premature collapse.

Model selection (spec §8.6): scored on a CRN seed block **disjoint** from the
reporting block ``0…8191``, so the shipped checkpoint is not selected on the
seeds that later report it. ``{scenario}_ppo.zip`` is that selection artifact;
``{scenario}_ppo_final.zip`` is the terminal one. They are different networks —
always say which produced a number.

Usage:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python clark_scarf_ppo_train.py \
        -s n3_l2_p09 -o raw --level L1
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from clark_scarf_eval_common import eval_seed_block, rollout_seeds
from clark_scarf_gym import ClarkScarfEnv
from clark_scarf_scenarios import SCENARIOS

BETA = 0.95
# selection seeds live ABOVE the reporting block 0…8191 (spec §8.6)
SELECT_SEED_OFFSET = 1_000_000


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train PPO on clark_scarf.")
    p.add_argument("-s", "--scenario_name", default="n3_l2_p09",
                   choices=list(SCENARIOS.keys()))
    p.add_argument("-o", "--observation_mode", default="raw",
                   choices=list(ClarkScarfEnv.OBS_MODES))
    p.add_argument("-a", "--action_mode", default="ship_discrete",
                   choices=list(ClarkScarfEnv.ACTION_MODES))
    p.add_argument("--level", default="L1", choices=["L0", "L1"],
                   help="L1 = the §8.6 derivation (default); L0 = library "
                        "defaults + gamma=beta, no VecNormalize (control only)")
    p.add_argument("--total-timesteps", type=int, default=2_000_000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--outdir", default="results")
    p.add_argument("--report-every", type=int, default=50_000)
    p.add_argument("--select-every", type=int, default=100_000)
    p.add_argument("--select-n-seeds", type=int, default=256)
    # --learning_rate dest is the mdp_tuning contract (spec §8.2)
    p.add_argument("--learning_rate", "--lr-init", dest="learning_rate",
                   type=float, default=3e-4)
    p.add_argument("--lr-final", type=float, default=3e-5)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--gamma", type=float, default=BETA)   # defaults to beta; tunable
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=0.005)
    # dests match the mdp_tuning knob names (spec §8.2) so the tuner can reach
    # them; clip_init/max_grad_norm/vf_coef are out of the `breadth` tier but
    # open under --knobs all
    p.add_argument("--clip-init", "--clip-range", dest="clip_init",
                   type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--net_arch", type=int, nargs="+", default=[64, 64])
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--mask", action="store_true",
                   help="L2(gym) escalation #2: MaskablePPO restricted to "
                        "feasible shipment quantities (ship_discrete only)")
    p.add_argument("--ent-final", type=float, default=None,
                   help="L2(hp) escalation #4: linearly decay ent_coef to this "
                        "value; default None keeps it constant")
    p.add_argument("--no-norm-obs", action="store_false", dest="norm_obs", default=True)
    p.add_argument("--no-norm-reward", action="store_false", dest="norm_reward",
                   default=True)
    return p


def parse_args() -> argparse.Namespace:
    args = _build_arg_parser().parse_args()
    if args.level == "L0":
        # faithful library defaults; gamma stays at beta because that is the
        # spec's faithful STARTING point for a control run, not because gamma
        # is untunable — it is a training knob (spec §8.2)
        args.learning_rate = 3e-4
        args.lr_final = 3e-4          # constant, no schedule
        args.n_steps = 2048
        args.batch_size = 64
        args.n_epochs = 10
        args.gae_lambda = 0.95
        args.ent_coef = 0.0
        args.clip_init = 0.2
        args.net_arch = [64, 64]
        args.n_envs = 1
        args.norm_obs = False
        args.norm_reward = False
    return args


def linear_schedule(start: float, end: float) -> Callable[[float], float]:
    def _fn(progress_remaining: float) -> float:
        return end + progress_remaining * (start - end)
    return _fn


def build_schedule(start: float, end: float) -> float | Callable[[float], float]:
    return start if start == end else linear_schedule(start, end)


_SHORT = {"observation_mode": "obs", "action_mode": "act", "level": "lvl",
          "total_timesteps": "steps", "seed": "seed", "learning_rate": "lr",
          "lr_final": "lrf", "net_arch": "arch", "n_envs": "nenvs",
          "n_steps": "nsteps", "batch_size": "bs", "n_epochs": "ep",
          "gamma": "gamma", "gae_lambda": "gae", "ent_coef": "ent",
          "clip_init": "clip", "vf_coef": "vf", "max_grad_norm": "grad", "norm_obs": "normobs", "norm_reward": "normrew"}
_SKIP = {"outdir", "scenario_name", "report_every", "select_every",
         "select_n_seeds"}
_SHORT.update({"mask": "mask", "ent_final": "entf"})


def build_run_name(args: argparse.Namespace) -> str:
    """Encode obs/act/level plus every non-default hyperparameter (spec §8.4)."""
    defaults = vars(_build_arg_parser().parse_args([]))
    parts = [f"obs{args.observation_mode}", f"lvl{args.level}"]
    for key, dflt in defaults.items():
        if key in _SKIP or key in ("observation_mode", "level"):
            continue
        val = getattr(args, key)
        if val != dflt:
            lbl = _SHORT.get(key, key)
            parts.append(f"{lbl}{val:.3g}" if isinstance(val, float) else f"{lbl}{val}")
    return f"PPO_{datetime.now().strftime('%Y%m%d_%H%M%S')}_" + "_".join(parts)


def resolve_paths(outdir_arg: str, scenario: str, run_name: str) -> Path:
    outdir = (Path(__file__).resolve().parent / outdir_arg / scenario / run_name).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


class _Tee:
    def __init__(self, stream, fh):
        self._stream, self._fh = stream, fh

    def write(self, data: str) -> int:
        self._stream.write(data)
        self._fh.write(data)
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        self._fh.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def tee_console(outdir: Path, name: str = "train.log") -> None:
    """Mirror stdout/stderr into ``outdir/train.log`` (spec §8.4).

    The run directory only exists once the timestamped name is built, so a
    launcher cannot redirect into it — the script captures its own console.
    """
    fh = open(outdir / name, "a", buffering=1)
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)


def write_args(args: argparse.Namespace, outdir: Path) -> None:
    with open(outdir / f"{args.scenario_name}_ppo_args.txt", "w") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")


class CRNSelectionCallback(BaseCallback):
    """Select on a CRN block disjoint from the reporting seeds (spec §8.6).

    Scores the live policy with the *same* code the reporting eval uses, so the
    selection number and the reported number are commensurable — but on seeds
    the report never touches, so the shipped checkpoint is not chosen on the
    seeds that later judge it.
    """

    def __init__(self, scenario, obs_mode: str, act_mode: str, every: int,
                 n_seeds: int, train_env: VecNormalize, outdir: Path, tag: str,
                 masked: bool = False):
        super().__init__()
        self.scenario, self.obs_mode, self.act_mode = scenario, obs_mode, act_mode
        self.masked = masked
        self.every, self.n_seeds = every, n_seeds
        self.train_env, self.outdir, self.tag = train_env, outdir, tag
        self.best = np.inf
        self._next = every

    def _score(self) -> float:
        tmp = self.outdir / "_select_vecnormalize.pkl"
        self.train_env.save(str(tmp))
        vn = VecNormalize.load(str(tmp), DummyVecEnv([lambda: ClarkScarfEnv(
            self.scenario, observation_mode=self.obs_mode,
            action_mode=self.act_mode)]))
        vn.training = False

        def act_fn(obs, envs):
            o = vn.normalize_obs(obs) if self.train_env.norm_obs else obs
            if self.masked:
                masks = np.stack([e.action_masks() for e in envs])
                a, _ = self.model.predict(o, deterministic=True, action_masks=masks)
            else:
                a, _ = self.model.predict(o, deterministic=True)
            return a

        per = rollout_seeds(
            scenario=self.scenario, observation_mode=self.obs_mode,
            seeds=eval_seed_block(self.n_seeds, offset=SELECT_SEED_OFFSET),
            act_fn=act_fn, batch=64, progress_every=0,
            action_mode=self.act_mode)
        return float(per["cost_total"].mean())

    def _on_step(self) -> bool:
        if self.every and self.num_timesteps >= self._next:
            self._next += self.every
            score = self._score()
            better = score < self.best
            if better:
                self.best = score
                self.model.save(str(self.outdir / f"{self.tag}_ppo.zip"))
                self.train_env.save(str(self.outdir / "vecnormalize.pkl"))
            print(f"[select] step={self.num_timesteps} cost={score:.2f} "
                  f"best={self.best:.2f}{'  <- saved' if better else ''}", flush=True)
        return True


class EntropyDecayCallback(BaseCallback):
    """Linearly decay ent_coef (L2(hp) escalation #4).

    Rationale (#E4): with ``norm_reward=True`` returns are rescaled to ~unit
    variance while ``ent_coef`` stays fixed. Cost falls ~20x over training, so
    after normalization the entropy bonus grows RELATIVELY more important as the
    policy converges — which is the suspected cause of both L1 runs degrading
    ~5-8% after their 1.1M-step optimum. Decaying it removes that drift.
    """

    def __init__(self, start: float, end: float, total: int):
        super().__init__()
        self.start, self.end, self.total = start, end, total

    def _on_step(self) -> bool:
        frac = min(1.0, self.num_timesteps / max(1, self.total))
        self.model.ent_coef = self.start + frac * (self.end - self.start)
        return True


class ProgressCallback(BaseCallback):
    def __init__(self, every: int):
        super().__init__()
        self.every, self._next = every, every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._next += self.every
            ep = self.model.ep_info_buffer
            if ep:
                r = float(np.mean([e["r"] for e in ep]))
                print(f"[train] step={self.num_timesteps} "
                      f"ep_rew_mean={r:.2f} (cost {-r:.2f})", flush=True)
        return True


def build_training_env(args, scenario, outdir: Path) -> VecNormalize:
    def make_env(rank: int):
        def _make():
            env = ClarkScarfEnv(scenario, observation_mode=args.observation_mode,
                                action_mode=args.action_mode)
            return Monitor(env, filename=str(outdir / f"monitor_{rank}"))
        return _make

    venv = DummyVecEnv([make_env(i) for i in range(args.n_envs)])
    # with both flags off VecNormalize is a pass-through — that is how L0
    # expresses "no VecNormalize" without a second env-build path
    return VecNormalize(venv, norm_obs=args.norm_obs, norm_reward=args.norm_reward,
                        gamma=args.gamma, clip_obs=10.0)


def main() -> None:
    args = parse_args()
    scenario = SCENARIOS[args.scenario_name]
    run_name = build_run_name(args)
    outdir = resolve_paths(args.outdir, args.scenario_name, run_name)
    tee_console(outdir)
    print(f"outdir={outdir}")
    write_args(args, outdir)
    print(f"[cfg] level={args.level} obs={args.observation_mode} "
          f"N={scenario.n_echelons} L={scenario.leadtime} T={scenario.horizon} "
          f"gamma={args.gamma} (=beta) norm_obs={args.norm_obs}", flush=True)

    env = build_training_env(args, scenario, outdir)
    if args.mask:
        assert args.action_mode in ("ship_discrete", "target_discrete"), (
            "--mask needs a discrete encoding: masking a continuous Box is "
            f"meaningless, got {args.action_mode!r}"
        )
        print("[cfg] MaskablePPO: head restricted to feasible quantities", flush=True)
    algo = MaskablePPO if args.mask else PPO
    model = algo(
        "MlpPolicy", env,
        learning_rate=build_schedule(args.learning_rate, args.lr_final),
        n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.n_epochs,
        gamma=args.gamma, gae_lambda=args.gae_lambda, ent_coef=args.ent_coef,
        clip_range=args.clip_init,
        vf_coef=args.vf_coef, max_grad_norm=args.max_grad_norm, policy_kwargs={"net_arch": list(args.net_arch)},
        tensorboard_log=str(outdir), seed=args.seed, verbose=0,
        # MLP policy: SB3 itself warns the GPU is counterproductive here, and
        # pinning CPU keeps the four parallel runs from contending for one device
        device="cpu",
    )
    select_cb = CRNSelectionCallback(
        scenario, args.observation_mode, args.action_mode, args.select_every,
        args.select_n_seeds, env, outdir, args.scenario_name, masked=args.mask)
    cbs = [ProgressCallback(args.report_every), select_cb]
    if args.ent_final is not None:
        cbs.append(EntropyDecayCallback(args.ent_coef, args.ent_final,
                                        args.total_timesteps))
    model.learn(total_timesteps=args.total_timesteps, callback=cbs)
    model.save(str(outdir / f"{args.scenario_name}_ppo_final.zip"))
    env.save(str(outdir / "vecnormalize_final.pkl"))
    print(f"[done] selection artifact best cost = {select_cb.best:.2f}")
    print(f"[done] {outdir}")


if __name__ == "__main__":
    main()

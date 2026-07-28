"""PPO search space: tiers, β-aware gamma, rollout-aware bounds, and the
encode (warm-start) inverse.

Self-contained (optuna's FixedTrial is the only input), so it lives with the
other engine tests rather than inside the shipped package.
"""

from __future__ import annotations

from optuna.trial import FixedTrial

from mdp_tuning.spaces import (
    PPO_KNOBS, PPO_TIER_BREADTH, PPO_TIER_CORE, SPACES, encode_ppo,
    n_steps_log2_low, sample_ppo,
)


def test_tiers_nest() -> None:
    assert set(PPO_TIER_CORE) < set(PPO_TIER_BREADTH) < set(PPO_KNOBS)
    space = SPACES["ppo"]
    assert space.tiers["core"] == PPO_TIER_CORE
    assert space.tiers["all"] == PPO_KNOBS
    # the dest-mismatch fix: the space speaks the script's schedule dest
    assert "clip_init" in PPO_KNOBS and "clip_range" not in PPO_KNOBS
    # frozen tier stays out of core/breadth
    for knob in ("clip_init", "max_grad_norm", "gamma"):
        assert knob not in PPO_TIER_BREADTH


def test_gamma_containment() -> None:
    # β itself must be reachable (center-containment rule)
    cfg = sample_ppo(FixedTrial({"gamma_at_beta": True}), {"gamma"}, beta=1.0)
    assert cfg["gamma"] == 1.0
    cfg = sample_ppo(FixedTrial({"gamma_at_beta": True}), {"gamma"}, beta=0.95)
    assert cfg["gamma"] == 0.95
    # below-β draws never exceed β
    cfg = sample_ppo(FixedTrial({"gamma_at_beta": False,
                                 "beta_minus_gamma": 0.001}),
                     {"gamma"}, beta=0.95)
    assert abs(cfg["gamma"] - 0.949) < 1e-9


def test_batch_cap_respects_n_envs() -> None:
    trial_params = {"log2_n_steps": 8, "log2_batch_size": 9}
    single = sample_ppo(FixedTrial(trial_params), {"n_steps", "batch_size"},
                        n_envs=1)
    assert single["batch_size"] == 256          # capped at n_steps
    multi = sample_ppo(FixedTrial(trial_params), {"n_steps", "batch_size"},
                       n_envs=4)
    assert multi["batch_size"] == 512           # rollout = 256*4 admits 512


def test_n_steps_floor() -> None:
    assert n_steps_log2_low(None) == 8
    assert n_steps_log2_low(2048) == 11
    assert n_steps_log2_low(2500) == 12
    assert n_steps_log2_low(10_000_000) == 12   # capped at the space max


def test_derived_one_dof() -> None:
    derived = SPACES["ppo"].derived
    src, factor = derived["lr_final"]
    assert src == "learning_rate" and factor == 0.1
    src, factor = derived["clip_final"]
    assert src == "clip_init" and factor == 0.25


def test_encode_roundtrip() -> None:
    defaults = {
        "learning_rate": 3e-4, "ent_coef": 0.005, "gamma": 1.0,
        "gae_lambda": 0.95, "clip_init": 0.2, "n_steps": 2048,
        "n_epochs": 10, "vf_coef": 0.5, "max_grad_norm": 0.5,
        "net_arch": (64, 64), "batch_size": 256,
    }
    params, skipped = encode_ppo(defaults, beta=1.0)
    assert not skipped, skipped
    cfg = sample_ppo(FixedTrial(params), set(defaults), beta=1.0)
    for k, v in defaults.items():
        got = cfg[k]
        if isinstance(v, tuple):
            assert tuple(got) == v, (k, got, v)
        else:
            assert abs(float(got) - float(v)) < 1e-9, (k, got, v)


def test_encode_skips_unrepresentable() -> None:
    params, skipped = encode_ppo(
        {"ent_coef": 0.0,            # outside the log range
         "n_steps": 3000,            # not a power of two
         "net_arch": (17, 3),        # no label
         "learning_rate": 1e-4},
        beta=1.0)
    assert set(skipped) == {"ent_coef", "n_steps", "net_arch"}
    assert params == {"learning_rate": 1e-4}
    # a default below a raised n_steps floor is skipped, not enqueued stale
    params, skipped = encode_ppo({"n_steps": 2048}, min_n_steps=2500)
    assert skipped == ["n_steps"] and not params

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
    # frozen tier: what stays at its derived value unless explicitly opened
    for knob in ("clip_init", "max_grad_norm"):
        assert knob not in PPO_TIER_BREADTH
    # core corrects what the L1 derivation actually guesses at — gae_lambda is
    # per-instance by §8.6 and two campaigns bracket its optimum ~10x apart
    assert "gae_lambda" in PPO_TIER_CORE
    # gamma is bounded by the problem (sampler cannot exceed β), so it opens a
    # tier earlier than the genuinely frozen knobs
    assert "gamma" in PPO_TIER_BREADTH and "gamma" not in PPO_TIER_CORE


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


def test_long_horizon_lambda_is_inside_the_space() -> None:
    """Upstream #4: the old 1−λ floor of 0.01 capped the credit horizon at
    100 steps — mab's hand-set λ=0.994 was unreachable and could not even be
    enqueued as the warm-start trial. The range stays ABSOLUTE (never coupled
    to T̄): two campaigns measured opposite optima, so the tuner searches the
    full range and no episode-relative floor is imposed."""
    params, skipped = encode_ppo({"gae_lambda": 0.994}, beta=1.0)
    assert not skipped
    cfg = sample_ppo(FixedTrial(params), {"gae_lambda"}, beta=1.0)
    assert abs(cfg["gae_lambda"] - 0.994) < 1e-9

    # both campaigns' winners representable: game2048's 0.90 and λ→0.9995
    for lam in (0.90, 0.9995):
        params, skipped = encode_ppo({"gae_lambda": lam}, beta=1.0)
        assert not skipped, lam


# --- net_arch as two ordered axes (width in core, depth at breadth) --------

def _arch(params: dict, tier: str, depth_default: int = 2) -> tuple[int, ...]:
    cfg = sample_ppo(FixedTrial(params), {"net_arch"}, tier=tier,
                     net_depth_default=depth_default)
    return tuple(cfg["net_arch"])


def test_core_searches_width_and_holds_the_derived_depth() -> None:
    """§8.6 derives a width from obs dim and says nothing about layer count, so
    core corrects the number the derivation produced and leaves the other."""
    assert _arch({"log2_net_width": 8}, "core") == (256, 256)
    assert _arch({"log2_net_width": 5}, "core", depth_default=3) == (32, 32, 32)


def test_breadth_opens_depth() -> None:
    assert _arch({"log2_net_width": 7, "net_depth": 4}, "breadth") == (128,) * 4
    assert _arch({"log2_net_width": 6, "net_depth": 3}, "all") == (64, 64, 64)


def test_the_grid_is_four_widths_by_three_depths() -> None:
    shapes = {_arch({"log2_net_width": w, "net_depth": d}, "all")
              for w in range(5, 9) for d in range(2, 5)}
    assert len(shapes) == 12
    assert (64, 64) in shapes and (128, 128, 128) in shapes and (256, 256) in shapes


def test_the_shape_a_script_defaults_to_is_encodable() -> None:
    """upstream #56: a (128,128) default matched no label, so the knob was
    dropped from the warm start and trial 0 was not the L1 centre."""
    for shape in [(64, 64), (128, 128), (256, 256), (32, 32, 32), (128, 128, 128)]:
        params, skipped = encode_ppo({"net_arch": shape}, tier="all")
        assert not skipped, shape
        assert _arch(params, "all") == shape


def test_a_non_uniform_shape_is_skipped_not_guessed() -> None:
    _, skipped = encode_ppo({"net_arch": (400, 300)}, tier="all")
    assert skipped == ["net_arch"]


def test_core_encodes_a_depth_it_does_not_search() -> None:
    """Refusing to encode an out-of-range depth at core would strand the warm
    start over an axis core never touches."""
    params, skipped = encode_ppo({"net_arch": (64,) * 6}, tier="core")
    assert not skipped and params == {"log2_net_width": 6}


def test_a_study_that_recorded_the_old_categorical_still_resumes(tmp_path) -> None:
    """The reason net_arch became two int axes rather than a longer list.

    Optuna keys a categorical by its ordered choices tuple, so *growing* the old
    four-label list raises "CategoricalDistribution does not support dynamic
    value space" on the first suggest of every study that already recorded it —
    every downstream study, stranded. Retiring the param instead leaves the old
    trials readable and simply stops suggesting it.
    """
    import logging
    import optuna

    optuna.logging.set_verbosity(logging.CRITICAL)
    storage = f"sqlite:///{tmp_path / 'optuna.db'}"

    def old(trial):
        trial.suggest_categorical("net_arch", ["small", "medium", "large", "deep"])
        return 0.0

    study = optuna.create_study(study_name="s", storage=storage)
    study.optimize(old, n_trials=2)

    study = optuna.create_study(study_name="s", storage=storage, load_if_exists=True)
    study.optimize(lambda t: (sample_ppo(t, {"net_arch"}, tier="all"), 0.0)[1],
                   n_trials=2)

    assert len(study.trials) == 4
    assert {tuple(sorted(t.params)) for t in study.trials} == {
        ("net_arch",), ("log2_net_width", "net_depth")}

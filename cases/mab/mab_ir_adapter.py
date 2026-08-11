"""Differential adapter for the ``mab`` domain (portable-domain contract:
lives with the domain, discovered from the IR file's directory by
``mdp_ir.differential.load_adapter_factory``)."""

from __future__ import annotations

from pathlib import Path

from mdp_ir.differential import DomainAdapter, _import_domain
from mdp_ir.schema import MdpIR


def make_adapter(
    ir: MdpIR,
    instance: str | None = None,
    seed_salt: int = 0,
    domain_dir: Path | None = None,
) -> DomainAdapter:
    """Builds the selected branch's scenario source from the resolved IR and
    drives the ``init_state`` → ``advance`` loop.

    The IR is a catalog: its one ``payout`` slot resolves to a single source
    whose family names the selected candidate, and the candidate's per-seed
    ``iid`` latent desugars into the synthesized ``payout_arm_means`` constant
    drawn by the ``payout_latent`` sampler. The domain's latent generator owns
    the same recipe at the same source id, so both sides realize the arm means
    bit-for-bit.
    """
    unc, scen, mdp = _import_domain(
        domain_dir or Path(__file__).resolve().parent,
        ["mab_uncertainty", "mab_scenarios", "mab_mdp"],
    )

    # instance-resolved constants: the base values with the named instance's
    # overrides applied, mirroring MdpBlock.horizon_T. #E32 promoted n_arms /
    # arm_max / horizon_T to axis-tagged constants, so K, T and sigma all vary
    # per instance and reading the base values alone would silently build the
    # wrong domain (idempotent if the loader already resolved them).
    consts = {c.name: c.value for c in ir.mdp.scenario.constants}
    if instance:
        consts.update(ir.mdp.scenario.instances[instance])
    payout_src = next(s for s in ir.mdp.uncertainty_sources if s.name == "payout")
    # from the constant, not len(payout_arm_means): the synthesized latent's
    # value is the static *example*, whose length is fixed at the base K
    n_arms = int(consts["n_arms"])
    source_id = payout_src.stream_id

    # the resolved family names the selected candidate (bernoulli | normal)
    if payout_src.distribution.family == "bernoulli":
        latent = unc.LatentBernoulliPayout(
            n_arms=n_arms, low=0.0, high=1.0, source_id=source_id
        )
    else:
        latent = unc.LatentGaussianPayout(
            n_arms=n_arms, prior_mean=0.0, prior_sd=1.0,
            sigma=consts["sigma"], source_id=source_id,
        )

    template = scen.MabScenario(
        scenario_name=f"ir_differential_{instance or 'base'}",
        horizon=ir.mdp.horizon_T(instance),
        payout=latent,
        seed_salt=seed_salt,
    )
    source = scen.MabScenarioSource(template)

    def run_episode(episode_seed: int, decisions: list[dict]) -> list[dict]:
        scenario = source(episode_seed)
        state, _ = mdp.init_state(scenario, episode_seed)
        rows: list[dict] = []
        for acts in decisions:
            if state.terminated:
                break
            state, info = mdp.advance(scenario, state, arm=int(acts["arm"]))
            rows.append({
                "t": info["action_period"],
                "arm": info["arm"],
                "reward": info["payout"],
                "mean_pulled": info["mean_pulled"],
                "opt_mean": info["opt_mean"],
                "pulls": list(state.pulls),
                "payouts": list(state.payouts),
                **info["payoff"],   # pull, total
            })
        return rows

    return run_episode

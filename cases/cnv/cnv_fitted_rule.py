"""The fitted structural rule, scored (spec §14.2): order = y^M(posterior) + premium.

Proposition 1(b)'s structure is an additive premium over the myopic newsvendor
level of the policy's own observation type. The premium is read from a probe's
on-policy dump on the SCREEN block (``cnv_policy_probe.py`` → ``onpolicy_{target}.tsv``,
never the protocol block) as the rounded mean excess ``y_net − y^M``:

- ``--form cell_period``: one premium per (cell, period) — the net's structure
  read cell by cell;
- ``--form period``: one premium per period for the whole grid — the compact
  rule (N numbers per type).

The rule is then a policy like any other arm: scored under the §9 protocol
(same seed block, CRN), written as a spec-§9 TSV + per-seed sidecar, and
paired against the net it was read from and the type's bar. The verdict
branches are pre-decided (spec §14.2): |fitted − net| ≤ 1% of the bar → the
net implements the structure; > 1% → the net is doing something else; the
fitted rule beating its net makes the rule a shippable artifact.

    python cnv_fitted_rule.py --probe results/tuning/.../probe/onpolicy_paper510.tsv -g paper510 \\
        --form cell_period --n-seeds 2048 --net-seeds results/tuning/<study>/winner_protocol_paper510.seeds.tsv \\
        --bar-seeds results/paper510/benchmark/benchmark_dp_timing_eval_paper510.seeds.tsv --name tb_ctx_cell
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from cnv_benchmark_common import (HEADER, METRICS, build_arg_parser, eval_seed_block, eval_targets, format_row,
                                  rollout, summarize)
from cnv_policy_probe import Reference, paired_delta

HERE = Path(__file__).resolve().parent


def read_probe(path: Path) -> tuple[str, list[dict]]:
    """``(observation type, rows)`` from an on-policy dump; the type is stamped in its header."""
    kind, rows, hdr = None, [], None
    for ln in Path(path).read_text().splitlines():
        if ln.startswith("#"):
            if " type=" in ln:
                kind = ln.split(" type=")[1].split()[0]
            continue
        f = ln.split("\t")
        if hdr is None:
            hdr = f
            continue
        r = dict(zip(hdr, f))
        rows.append({"target": r["target"], "period": int(r["period"]),
                     "excess": int(r["y_net"]) - int(r["y_myopic"])})
    assert kind in ("timing", "full", "event"), f"{path}: no observation type stamped"
    return kind, rows


def fit_premium(rows: list[dict], form: str, horizon: int, targets: list[str]) -> dict:
    """Rounded mean excess per (cell, period) or per period; asserts the fit covers every period (§14.2)."""
    acc = defaultdict(list)
    for r in rows:
        acc[(r["target"], r["period"]) if form == "cell_period" else r["period"]].append(r["excess"])
    prem = {k: int(np.round(np.mean(v))) for k, v in acc.items()}
    if form == "cell_period":
        missing = [(t, p) for t in targets for p in range(horizon) if (t, p) not in prem]
    else:
        missing = [p for p in range(horizon) if p not in prem]
    assert not missing, f"the fit does not cover the horizon (first gaps: {missing[:5]}) — refusing a partial rule"
    return prem


def make_rule(kind: str, prem: dict, form: str, row_id: str):
    ref, holder = Reference(kind), {"seed": None}

    def act(env, obs=None) -> int:
        if holder["seed"] != env._episode_seed:      # a new episode: the event type's posterior restarts
            holder["seed"] = env._episode_seed
            ref.new_episode(env)
        y_m, _ = ref(env)
        p = env._state.period
        return max(0, y_m + prem[(row_id, p) if form == "cell_period" else p])
    return act


def main() -> None:
    p = build_arg_parser("Score the fitted premium rule (spec §14.2).")
    p.add_argument("--probe", type=str, required=True, help="the net's on-policy dump on the screen block")
    p.add_argument("--form", choices=("cell_period", "period"), default="cell_period")
    p.add_argument("--name", type=str, required=True, help="the arm's name in the output file")
    p.add_argument("--net-seeds", type=str, default=None, help="the net's protocol per-seed sidecar, for the pair")
    p.add_argument("--bar-seeds", type=str, default=None, help="the type's bar per-seed sidecar, for the pair")
    args = p.parse_args()
    if args.first_seed != 0:
        print("[fitted] WARNING: not the protocol block — never quote this number (§9.7)")

    kind, rows = read_probe(Path(args.probe))
    target, cells = eval_targets(args)
    horizon = int(cells[0][1](0).horizon)
    prem = fit_premium(rows, args.form, horizon, [c for c, _ in cells])
    seeds = eval_seed_block(args.first_seed, args.n_seeds)
    outfile = Path(args.outfile) if args.outfile else HERE / args.outdir / target / "benchmark" / f"benchmark_fitted_{args.name}_eval_{target}.tsv"
    outfile.parent.mkdir(parents=True, exist_ok=True)
    seeds_file = outfile.with_name(outfile.stem + ".seeds.tsv")
    if args.form == "period":
        print("[fitted] premium by period:", {k: prem[k] for k in sorted(prem)})
    stamps = [f"# benchmark: fitted_{args.name} (order = y^M_{kind} + premium, form={args.form})",
              f"# probe: {args.probe}", f"# target: {target}",
              f"# seed block: first_seed={args.first_seed} n_seeds={args.n_seeds}",
              f"# observation_mode={args.observation_mode} action_mode={args.action_mode}"]
    with open(outfile, "w") as f, open(seeds_file, "w") as fs:
        f.write("\n".join(stamps) + "\n" + "\t".join(HEADER) + "\n"); f.flush()
        fs.write("target\tseed\t" + "\t".join(METRICS) + "\n")
        for row_id, source in cells:
            per_seed = rollout(source, seeds, make_rule(kind, prem, args.form, row_id),
                               observation_mode=args.observation_mode, action_mode=args.action_mode,
                               progress_every=0)
            f.write(format_row(row_id, source, len(seeds), summarize(per_seed)) + "\n"); f.flush()
            for k, seed in enumerate(seeds):
                fs.write(f"{row_id}\t{seed}\t" + "\t".join(f"{per_seed[m][k]:.6f}" for m in METRICS) + "\n")
    print(f"[fitted] record -> {outfile}")
    for label, other in (("fitted − net", args.net_seeds), ("fitted − bar", args.bar_seeds)):
        if other:
            res = paired_delta(Path(other), seeds_file)
            print(f"[fitted] {label}: {res['delta_mean']:+.4f} ± {res['delta_se']:.4f} over {res['n']} (cell, seed) pairs"
                  f"   [{Path(other).name} {res['a_mean']:.4f} · fitted {res['b_mean']:.4f}]")


if __name__ == "__main__":
    main()

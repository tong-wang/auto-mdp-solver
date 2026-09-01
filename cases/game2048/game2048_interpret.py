"""Interpretation ladder for a trained game2048 policy (pilot for the skill's
third pillar: interpret the learned policy back into structural insight).

PPO ships two artifacts, not one: the actor answers "what would you do", the
critic answers "how good is this position" — and most of the ladder below
reads the CRITIC, which classical analysis never has access to at this
fidelity. Stages:

  replay          rung 0: annotated GIF/text of sample paths — board, the
                  actor's action distribution, the critic's V(s), per step
  structure       rung 1: structure metrics measured DURING LIFE (corner
                  occupancy + which corner, episode-max snake chain, monotone
                  lines) over deterministic rollouts; --with-greedy appends a
                  myopic-baseline row through the same instrument. Committed
                  2026-08-04 (0c); the pilot's rung-1 numbers came from an
                  uncommitted ad-hoc of exactly this computation
  probes          rung 2+3: value calibration, death anatomy, occlusion
                  saliency, actor-vs-critic-greedy consistency, and the
                  critic regression on structure features
  counterfactual  rung 3: common-random-number action branches — exact
                  score-to-go per action under the policy's own continuation
                  (the keyed spawn streams make branches share randomness),
                  scored against both the actor's choice and the critic's
                  ranking
  symmetry        A9 rung 1: orbit probes against the certified D4 engine
                  fact (test 39) — actor orbit-agreement (the commitment's
                  footprint) and critic-vs-CRN-truth across orbits (does the
                  critic KNOW the commitment?)
  trunkprobe      the TP instrument (#E23's open "D3": can't-represent vs
                  never-learned-to-use): linear probes on frozen trunk
                  activations for per-cell relational primitives (neighbor
                  exponent diffs, mergeable-pair indicators, occupancy) and
                  board-level structure, against a random-init control trunk
                  and the input-patch baseline. Registered in ESCALATION.md
                  (TP entry) before its first full run
  frameavg        A9 rung 2: the crown symmetrized by frame averaging,
                  scored under the standard seed protocol (pre-registered
                  in ESCALATION.md A9 before the run)

Usage (from the domain folder, any venv with the [domain] extra):
  python game2048_interpret.py replay --model-path <run>/best_model.zip --seed 3
  python game2048_interpret.py probes --model-path ... --n-seeds 64
  python game2048_interpret.py counterfactual --model-path ... --n-states 60

Outputs land in <model dir>/interpret/ (folder-relative, portable-domain
contract). All numbers at small n are marked provisional by the report.
"""

from __future__ import annotations

import argparse
import pickle
from math import log2
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from game2048_board import apply_move, d4_transforms, slide_score, valid_moves
from game2048_gym import OBSERVATION_MODES
from game2048_mdp import advance, init_state, open_episode, valid_actions
from game2048_policy import Game2048Policy
from game2048_scenarios import SCENARIOS

MOVES = ["Up", "Down", "Right", "Left"]


# ---------------------------------------------------------------------------
# model access: the critic and the action distribution, in raw-score units
# ---------------------------------------------------------------------------

class PolicyLens:
    """Reads both PPO heads through the deployable wrapper's own encoder."""

    def __init__(self, model_path: str, grid_size: int = 3,
                 observation_mode: str = "onehot"):
        self.pol = Game2048Policy(model_path, observation_mode=observation_mode,
                                  grid_size=grid_size)
        self.grid_size = grid_size
        # VecNormalize trained with norm_reward=True: the critic predicts
        # NORMALIZED returns. ret_rms lets us report V in raw merge points.
        self.ret_scale = 1.0
        vn = Path(model_path).parent / "vecnormalize.pkl"
        if vn.exists():
            with open(vn, "rb") as fh:
                obj = pickle.load(fh)
            if getattr(obj, "norm_reward", False):
                self.ret_scale = float(np.sqrt(obj.ret_rms.var + 1e-8))

    def _tensor(self, boards: list[list[int]]) -> torch.Tensor:
        obs = np.stack([self.pol._encode(b) for b in boards])
        t, _ = self.pol.model.policy.obs_to_tensor(obs)
        return t

    def values(self, boards: list[list[int]]) -> np.ndarray:
        """V(s) in raw merge points, batched."""
        with torch.no_grad():
            v = self.pol.model.policy.predict_values(self._tensor(boards))
        return v.cpu().numpy().reshape(-1) * self.ret_scale

    def probs(self, board: list[int]) -> np.ndarray:
        mask = np.array(valid_moves(board), dtype=bool)[None]
        with torch.no_grad():
            dist = self.pol.model.policy.get_distribution(
                self._tensor([board]), action_masks=mask)
            p = dist.distribution.probs.cpu().numpy().reshape(-1)
        return p

    def probs_batch(self, boards: list[list[int]]) -> np.ndarray:
        """(k, 4) masked action probabilities, batched."""
        masks = np.array([valid_moves(b) for b in boards], dtype=bool)
        with torch.no_grad():
            dist = self.pol.model.policy.get_distribution(
                self._tensor(boards), action_masks=masks)
            p = dist.distribution.probs.cpu().numpy()
        return p

    def act(self, board: list[int]) -> int:
        return self.pol.act(board)


# ---------------------------------------------------------------------------
# structure features (rung-1 vocabulary, reused as the rung-3 dictionary)
# ---------------------------------------------------------------------------

def board_features(board: list[int], n: int) -> dict[str, float]:
    corners = {0, n - 1, n * (n - 1), n * n - 1}
    edges = {i for i in range(n * n)
             if i // n in (0, n - 1) or i % n in (0, n - 1)}
    m = max(board)
    g = [board[r * n:(r + 1) * n] for r in range(n)]
    lines = g + [list(c) for c in zip(*g)]
    mono = sum(l == sorted(l) or l == sorted(l, reverse=True) for l in lines)
    adj = sum(1 for r in range(n) for c in range(n - 1)
              if g[r][c] and g[r][c] == g[r][c + 1])
    adj += sum(1 for c in range(n) for r in range(n - 1)
               if g[r][c] and g[r][c] == g[r + 1][c])

    def nbrs(i):
        r, c = divmod(i, n)
        return [j for j in (i - n if r else None, i + n if r < n - 1 else None,
                            i - 1 if c else None, i + 1 if c < n - 1 else None)
                if j is not None]

    def chain(i):
        best = 1
        for j in nbrs(i):
            if board[j] * 2 == board[i] and board[j] > 0:
                best = max(best, 1 + chain(j))
        return best

    snake = max((chain(i) for i, v in enumerate(board) if v == m), default=0)
    return {
        "n_empty": sum(v == 0 for v in board),
        "k_max": 0 if m == 0 else log2(m),
        "corner_max": float(any(board[i] == m for i in corners)),
        "edge_max": float(any(board[i] == m for i in edges)),
        "mono_lines": float(mono),
        "adj_pairs": float(adj),
        "snake_len": float(snake),
    }


FEATURE_NAMES = ["n_empty", "k_max", "corner_max", "edge_max",
                 "mono_lines", "adj_pairs", "snake_len"]


# ---------------------------------------------------------------------------
# rollout collection (shared by every stage)
# ---------------------------------------------------------------------------

def rollout(lens: PolicyLens, scenario, seed: int) -> dict:
    st = open_episode(scenario, init_state(scenario, seed)[0])
    boards, actions, gains, probs_l = [], [], [], []
    while True:
        va = valid_actions(st)
        if not any(va) or st.period >= scenario.T_cap:
            break
        boards.append(list(st.board))
        p = lens.probs(st.board)
        a = lens.act(st.board)
        st, info = advance(scenario, st, a)
        actions.append(a)
        gains.append(info["gained"])
        probs_l.append(p)
    died = st.period < scenario.T_cap
    return {"seed": seed, "boards": boards, "actions": actions,
            "gains": np.array(gains), "probs": probs_l, "died": died,
            "final_board": list(st.board)}


# ---------------------------------------------------------------------------
# stage: replay (rung 0)
# ---------------------------------------------------------------------------

TILE_COLORS = {0: "#cdc1b4", 1: "#eee4da", 2: "#ede0c8", 3: "#f2b179",
               4: "#f59563", 5: "#f67c5f", 6: "#f65e3b", 7: "#edcf72",
               8: "#edcc61", 9: "#edc850", 10: "#edc22e",
               # 4x4 reaches 2048+; without these every big tile renders as
               # the same dark fallback and the endgame becomes unreadable
               11: "#e8b923", 12: "#d4a017", 13: "#b8860b", 14: "#8b6508",
               15: "#5c4405"}


def stage_replay(lens: PolicyLens, scenario, seed: int, out_dir: Path,
                 max_frames: int = 240) -> Path:
    """Annotated replay. The text log keeps EVERY step; the GIF is strided to
    at most ``max_frames``, because a 4x4 episode runs ~700-1700 moves and a
    frame-per-move GIF is neither legible nor a sane file size."""
    from PIL import Image

    ep = rollout(lens, scenario, seed)
    vs = lens.values(ep["boards"])
    n = lens.grid_size
    n_steps = len(ep["boards"])
    # score BEFORE each move, so a strided frame still shows the right total
    cum = np.concatenate([[0.0], np.cumsum(ep["gains"])])
    stride = max(1, -(-n_steps // max_frames))
    keep = set(range(0, n_steps, stride)) | {n_steps - 1}
    frames, txt = [], []
    for t, (board, a, p, v) in enumerate(
            zip(ep["boards"], ep["actions"], ep["probs"], vs)):
        score = cum[t]
        txt.append(f"t={t:3d} score={score:6.0f} V={v:7.1f} "
                   f"act={MOVES[a]:5s} p={np.array2string(p, precision=2)} "
                   f"board={board}")
        if t not in keep:
            continue
        fig, (ax_b, ax_p) = plt.subplots(
            1, 2, figsize=(6.4, 3.2), width_ratios=[1, 1])
        ax_b.set_xlim(0, n); ax_b.set_ylim(0, n)
        ax_b.set_xticks([]); ax_b.set_yticks([]); ax_b.set_aspect("equal")
        for i, val in enumerate(board):
            r, c = divmod(i, n)
            k = 0 if val == 0 else int(log2(val))
            ax_b.add_patch(plt.Rectangle((c, n - 1 - r), 1, 1,
                           facecolor=TILE_COLORS.get(k, "#3c3a32"),
                           edgecolor="#bbada0", lw=3))
            if val:
                ax_b.text(c + .5, n - 1 - r + .5, str(val), ha="center",
                          va="center", fontsize=16, fontweight="bold",
                          color="#776e65" if k < 3 else "white")
        ax_b.set_title(f"t={t}  score={score:.0f}", fontsize=10)
        colors = ["#4a90d9" if i == a else "#c9c2b8" for i in range(4)]
        ax_p.bar(MOVES, p, color=colors)
        ax_p.set_ylim(0, 1)
        ax_p.set_title(f"actor π(a|s)   critic V(s) = {v:.0f} pts", fontsize=10)
        ax_p.tick_params(labelsize=8)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)
    gif = out_dir / f"replay_seed{seed}.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:],
                   duration=350, loop=0)
    (out_dir / f"replay_seed{seed}.txt").write_text("\n".join(txt) + "\n")
    print(f"  seed {seed}: {n_steps} moves, final score {cum[-1]:.0f}, "
          f"died={ep['died']} -> {gif.name} "
          f"({len(frames)} frames, stride {stride})")
    return gif


# ---------------------------------------------------------------------------
# stage: probes (rung 2 + the rung-3 regression)
# ---------------------------------------------------------------------------

def _pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x, y):
    rank = lambda v: np.argsort(np.argsort(v)).astype(float)  # noqa: E731
    return _pearson(rank(np.asarray(x)), rank(np.asarray(y)))


def stage_probes(lens: PolicyLens, scenario, n_seeds: int, out_dir: Path) -> str:
    n = lens.grid_size
    eps = [rollout(lens, scenario, s) for s in range(n_seeds)]
    for ep in eps:
        ep["values"] = lens.values(ep["boards"]) if ep["boards"] else np.array([])
        ep["togo"] = ep["gains"][::-1].cumsum()[::-1]  # gamma = 1

    V = np.concatenate([ep["values"] for ep in eps])
    G = np.concatenate([ep["togo"] for ep in eps])
    frac = np.concatenate([np.linspace(0, 1, len(ep["boards"]))
                           for ep in eps if ep["boards"]])
    lines = [f"# Interpretation probes — {n}x{n} | {out_dir.parent.name}",
             f"provisional: {n_seeds} seeds, deterministic\n",
             "## Value calibration (critic vs realized score-to-go)",
             f"- overall: pearson r = {_pearson(V, G):.3f}, "
             f"spearman = {_spearman(V, G):.3f}  (n = {len(V)} states)"]
    for lo, hi, tag in [(0, .33, "early"), (.33, .66, "mid"), (.66, 1.01, "late")]:
        m = (frac >= lo) & (frac < hi)
        lines.append(f"- {tag:5s} phase: pearson r = {_pearson(V[m], G[m]):.3f}, "
                     f"spearman = {_spearman(V[m], G[m]):.3f}")
    bias = float(np.mean(V - G))
    lines.append(f"- bias: mean(V - realized) = {bias:+.1f} pts "
                 f"({'optimist' if bias > 0 else 'pessimist'})")

    # death anatomy: does the critic see the end coming?
    K = 15
    dead = [ep for ep in eps if ep["died"] and len(ep["values"]) > K]
    if dead:
        tail = np.stack([ep["values"][-K:] for ep in dead])
        ent = np.stack([
            [-(p[p > 0] * np.log(p[p > 0])).sum() for p in ep["probs"][-K:]]
            for ep in dead])
        drop = tail[:, -1].mean() - tail[:, 0].mean()
        lines += ["\n## Death anatomy (last 15 moves before game over)",
                  f"- episodes dying naturally: {len(dead)}/{n_seeds}",
                  f"- mean V: {tail[:, 0].mean():.0f} -> {tail[:, -1].mean():.0f} "
                  f"pts (drop {drop:+.0f}) — "
                  + ("the critic anticipates death" if drop < -50 else
                     "the critic is surprised by death"),
                  f"- policy entropy: {ent[:, 0].mean():.2f} -> "
                  f"{ent[:, -1].mean():.2f} nats over the same window"]
        fig, ax = plt.subplots(1, 2, figsize=(8, 3))
        ax[0].plot(range(-K, 0), tail.mean(0)); ax[0].set_title("mean V(s), last 15 moves")
        ax[1].plot(range(-K, 0), ent.mean(0), color="darkred"); ax[1].set_title("mean entropy")
        for a in ax: a.set_xlabel("moves before death")
        fig.tight_layout(); fig.savefig(out_dir / "death_anatomy.png", dpi=110)
        plt.close(fig)

    # occlusion: which tiles does the critic consult?
    rng = np.random.default_rng(0)
    pool = [(ep, t) for ep in eps for t in range(len(ep["boards"]))]
    picks = [pool[i] for i in rng.choice(len(pool), size=min(150, len(pool)),
                                         replace=False)]
    pos_delta = np.zeros((n, n)); pos_cnt = np.zeros((n, n))
    max_d, other_d = [], []
    for ep, t in picks:
        b = ep["boards"][t]
        base = lens.values([b])[0]
        occl, idxs = [], []
        for i, v in enumerate(b):
            if v:
                bb = list(b); bb[i] = 0
                occl.append(bb); idxs.append(i)
        if not occl:
            continue
        dv = np.abs(lens.values(occl) - base)
        m = max(b)
        for i, d in zip(idxs, dv):
            pos_delta[i // n, i % n] += d; pos_cnt[i // n, i % n] += 1
            (max_d if b[i] == m else other_d).append(d)
    heat = pos_delta / np.maximum(pos_cnt, 1)
    lines += ["\n## Occlusion saliency (delta V when one tile is deleted)",
              f"- occluding the MAX tile: mean |dV| = {np.mean(max_d):.0f} pts",
              f"- occluding any other tile: mean |dV| = {np.mean(other_d):.0f} pts "
              f"(ratio {np.mean(max_d) / max(np.mean(other_d), 1e-9):.1f}x)",
              "- positional heatmap: occlusion_heatmap.png"]
    fig, ax = plt.subplots(figsize=(3.4, 3))
    im = ax.imshow(heat, cmap="viridis")
    for r in range(n):
        for c in range(n):
            ax.text(c, r, f"{heat[r, c]:.0f}", ha="center", va="center",
                    color="white", fontsize=9)
    ax.set_title("mean |dV| by position"); fig.colorbar(im)
    fig.tight_layout(); fig.savefig(out_dir / "occlusion_heatmap.png", dpi=110)
    plt.close(fig)

    # actor vs critic-greedy: is the actor the critic's own argmax?
    agree_cg = agree_greedy = total = 0
    gap = []
    for ep, t in picks:
        b = ep["boards"][t]
        va = valid_moves(b)
        if sum(va) < 2:
            continue
        st = _state_at(scenario, ep["seed"], t)
        nxt, qhat = [], []
        for a, ok in enumerate(va):
            if not ok:
                continue
            s2, info = advance(scenario, st, a)
            nxt.append(a)
            qhat.append(info["gained"] + lens.values([list(s2.board)])[0])
        a_actor = ep["actions"][t]
        a_cg = nxt[int(np.argmax(qhat))]
        a_greedy = max((a for a, ok in enumerate(va) if ok),
                       key=lambda a: slide_score(b, a))
        agree_cg += a_actor == a_cg
        agree_greedy += a_actor == a_greedy
        gap.append(max(qhat) - qhat[nxt.index(a_actor)]
                   if a_actor in nxt else float("nan"))
        total += 1
    lines += ["\n## Actor vs critic-greedy (one-step lookahead on the critic, CRN spawns)",
              f"- actor == argmax_a [gained + V(s')]: {agree_cg}/{total} "
              f"({agree_cg / total:.1%}) — internal consistency of the two heads",
              f"- actor == myopic greedy:             {agree_greedy}/{total} "
              f"({agree_greedy / total:.1%}) — how often it just takes the points",
              f"- mean critic-regret of the actor's move when it differs: "
              f"{np.nanmean(gap):.1f} pts"]

    # critic regression on the structure dictionary
    F = np.array([[board_features(ep["boards"][t], n)[k] for k in FEATURE_NAMES]
                  for ep in eps for t in range(len(ep["boards"]))])
    mu, sd = F.mean(0), F.std(0) + 1e-9
    Fz = (F - mu) / sd
    X = np.column_stack([np.ones(len(Fz)), Fz])
    beta, *_ = np.linalg.lstsq(X, V, rcond=None)
    r2 = 1 - np.sum((V - X @ beta) ** 2) / np.sum((V - V.mean()) ** 2)
    lines += ["\n## Critic regression (V regressed on structure features, standardized)",
              f"- R^2 = {r2:.3f}  (n = {len(V)}) — how much of the learned value "
              "IS nameable structure",
              "| feature | weight (pts per sd) |", "|---|---|"]
    order = np.argsort(-np.abs(beta[1:]))
    for i in order:
        lines.append(f"| {FEATURE_NAMES[i]} | {beta[1 + i]:+.1f} |")

    report = "\n".join(lines) + "\n"
    (out_dir / "probes.md").write_text(report)
    return report


# ---------------------------------------------------------------------------
# stage: structure (rung 1) — structure measured DURING LIFE, never at death
# ---------------------------------------------------------------------------

class _GreedyLens:
    """Myopic point-grabber through the same rollout instrument (baseline
    row). Deterministic; ties break by action order, which manufactures a
    corner preference — label it as an artifact, never a finding."""

    def __init__(self, grid_size: int):
        self.grid_size = grid_size

    def probs(self, board: list[int]) -> np.ndarray:
        va = valid_moves(board)
        a = max((i for i, ok in enumerate(va) if ok),
                key=lambda i: slide_score(board, i))
        p = np.zeros(4)
        p[a] = 1.0
        return p

    def act(self, board: list[int]) -> int:
        return int(np.argmax(self.probs(board)))


def _structure_row(lens, scenario, n_seeds: int) -> dict:
    n = lens.grid_size
    corners = [(0, "TL"), (n - 1, "TR"), (n * (n - 1), "BL"), (n * n - 1, "BR")]
    per_corner = {tag: 0 for _, tag in corners}
    corner_steps = total = 0
    mono_life = 0.0
    snake_ep, mono_death, scores, moves = [], [], [], []
    for s in range(n_seeds):
        ep = rollout(lens, scenario, s)
        if not ep["boards"]:
            continue
        ep_snake = 0.0
        for b in ep["boards"]:
            f = board_features(b, n)
            m = max(b)
            for idx, tag in corners:
                if b[idx] == m:
                    per_corner[tag] += 1
            corner_steps += f["corner_max"]
            mono_life += f["mono_lines"]
            ep_snake = max(ep_snake, f["snake_len"])
            total += 1
        snake_ep.append(ep_snake)
        mono_death.append(board_features(ep["boards"][-1], n)["mono_lines"])
        scores.append(float(ep["gains"].sum()))
        moves.append(len(ep["boards"]))
    tot_c = max(sum(per_corner.values()), 1)
    return {"states": total, "score": float(np.mean(scores)),
            "score_se": float(np.std(scores) / np.sqrt(len(scores))),
            "moves": float(np.mean(moves)),
            "corner_occ": corner_steps / total,
            "by_corner": {t: c / tot_c for t, c in per_corner.items()},
            "snake": float(np.mean(snake_ep)),
            "mono_life": mono_life / total,
            "mono_death": float(np.mean(mono_death))}


def stage_structure(lens: PolicyLens, scenario, n_seeds: int, out_dir: Path,
                    with_greedy: bool = False) -> str:
    """Rung 1. Structure is measured DURING LIFE (pilot rule 1): occupancy
    rates over all visited states and episode-max chain length — the
    terminal-board version is kept only as the pilot-compat column, where
    early death makes sorted lines trivially likely."""
    n = lens.grid_size
    rows = [("policy", _structure_row(lens, scenario, n_seeds))]
    if with_greedy:
        rows.append(("greedy", _structure_row(_GreedyLens(n), scenario,
                                              n_seeds)))
    lines = [f"# Structure during life — {n}x{n} | {out_dir.parent.name}",
             f"provisional: {n_seeds} seeds, deterministic\n",
             "| row | score | moves | states | corner occ | home | "
             f"snake ep-max /{n * n} | mono life /{2 * n} | mono@death |",
             "|---|---|---|---|---|---|---|---|---|"]
    for tag, r in rows:
        home = max(r["by_corner"], key=r["by_corner"].get)
        lines.append(
            f"| {tag} | {r['score']:.1f} ± {r['score_se']:.1f} "
            f"| {r['moves']:.1f} | {r['states']} | {r['corner_occ']:.3f} "
            f"| {home} {r['by_corner'][home]:.3f} | {r['snake']:.2f} "
            f"| {r['mono_life']:.2f} | {r['mono_death']:.2f} |")
    lines.append("\nby-corner (policy): " + "  ".join(
        f"{t}={v:.3f}" for t, v in rows[0][1]["by_corner"].items()))
    report = "\n".join(lines) + "\n"
    (out_dir / "structure.md").write_text(report)
    return report


def _state_at(scenario, seed: int, t: int):
    """Deterministically re-derive the state at step t of the crown's own
    trajectory (rollouts are deterministic, spawns are keyed)."""
    st = open_episode(scenario, init_state(scenario, seed)[0])
    lens = _state_at.lens
    for _ in range(t):
        st, _ = advance(scenario, st, lens.act(list(st.board)))
    return st


# ---------------------------------------------------------------------------
# stage: trunkprobe (TP — #E23's open "D3" instrument, generalized)
# ---------------------------------------------------------------------------

class _RandomLens:
    """Uniform-over-valid roller for the off-policy state pool. Seeded per
    instance so the pool is deterministic given the seed list and rollout()'s
    fixed call order; ``probs`` reports the uniform law it actually plays."""

    def __init__(self, grid_size: int, seed: int):
        self.grid_size = grid_size
        self.rng = np.random.default_rng(seed)

    def probs(self, board: list[int]) -> np.ndarray:
        va = np.array(valid_moves(board), dtype=float)
        return va / va.sum()

    def act(self, board: list[int]) -> int:
        va = valid_moves(board)
        return int(self.rng.choice([i for i, ok in enumerate(va) if ok]))


_DIRS = [("up", -1, 0), ("down", 1, 0), ("left", 0, -1), ("right", 0, 1)]


def _cell_targets(board: list[int], n: int):
    """Per-cell relational primitives of the current board.

    diff[i, d] = k(i) - k(neighbor_d(i)) with k the tile exponent; valid only
    where BOTH cells are on-board and occupied (diff_mask) — an exponent
    difference against an empty cell or a wall is not a comparison. equal[i,
    d] = 1 iff both occupied with the same exponent (the mergeable pair, the
    merge builtin's own predicate); its rows require only the neighbor to be
    on-board (eq_mask), so "neighbor empty" is a legitimate negative."""
    k = np.array([0.0 if v == 0 else log2(v) for v in board]).reshape(n, n)
    occ = (k > 0)
    diff = np.zeros((n, n, 4)); diff_mask = np.zeros((n, n, 4), dtype=bool)
    equal = np.zeros((n, n, 4)); eq_mask = np.zeros((n, n, 4), dtype=bool)
    nbr_k = np.zeros((n, n, 4))
    for d, (_, dr, dc) in enumerate(_DIRS):
        r0, r1 = max(0, -dr), n - max(0, dr)
        c0, c1 = max(0, -dc), n - max(0, dc)
        src = (slice(r0, r1), slice(c0, c1))
        dst = (slice(r0 + dr, r1 + dr), slice(c0 + dc, c1 + dc))
        eq_mask[src[0], src[1], d] = True
        nbr_k[src[0], src[1], d] = k[dst]
        both = occ[src] & occ[dst]
        diff_mask[src[0], src[1], d] = both
        diff[src[0], src[1], d] = np.where(both, k[src] - k[dst], 0.0)
        equal[src[0], src[1], d] = (both & (k[src] == k[dst])).astype(float)
    nn_ = n * n
    return (diff.reshape(nn_, 4), diff_mask.reshape(nn_, 4),
            equal.reshape(nn_, 4), eq_mask.reshape(nn_, 4),
            occ.reshape(nn_).astype(float),
            nbr_k.reshape(nn_, 4), k.reshape(nn_))


def _trunk_acts(fe, obs_np: np.ndarray, chunk: int = 256):
    """Activations after each ReLU of ``fe.cnn`` (per-cell maps) plus the
    ``fe.linear`` features, SmallBoardCnn-family layout (.cnn + .linear).

    NOTE this restricts the ``trunkprobe`` stage to the ``small`` / ``embed``
    / ``conv3d`` extractors. The ``rowcol``, ``axial`` and ``attn-*`` arms
    carry a different internal layout and are refused by the assert below."""
    import torch.nn as nn
    assert hasattr(fe, "cnn") and hasattr(fe, "linear"), (
        f"trunkprobe reads the SmallBoardCnn family (.cnn/.linear); "
        f"got {type(fe).__name__}")
    per_layer: dict[str, list[np.ndarray]] = {}
    feats = []
    with torch.no_grad():
        for lo in range(0, obs_np.shape[0], chunk):
            x = torch.as_tensor(obs_np[lo:lo + chunk])
            li = 0
            for mod in fe.cnn:
                x = mod(x)
                if isinstance(mod, nn.ReLU) and x.dim() == 4:
                    li += 1
                    per_layer.setdefault(f"L{li}", []).append(
                        x.cpu().numpy())
            feats.append(fe.linear(x).cpu().numpy())
    return ({k: np.concatenate(v) for k, v in per_layer.items()},
            np.concatenate(feats))


def _reinit_trunk(fe, seed: int = 1234):
    """Architecture-matched random-init control: linear probes on random conv
    features are strong (the random-features kernel effect), so every trained
    number is read against this control, never alone."""
    import copy
    ctrl = copy.deepcopy(fe)
    torch.manual_seed(seed)
    for mod in ctrl.modules():
        if hasattr(mod, "reset_parameters"):
            mod.reset_parameters()
    return ctrl


def _ridge(Xtr, ytr, Xte, alphas):
    """Held-out scores of a ridge probe (columns standardized on train),
    strength selected on a 20% within-train validation split.

    Two instrument lessons are baked in (both caught by the self-gate,
    2026-08-25): float64 throughout, because float32 Gram roundoff cost
    0.003 R^2 on the 4x4 input-patch solve; and alpha SELECTED, not fixed,
    because on high-dimensional probe inputs a weak fixed alpha keeps
    spurious train-episode correlations whose weights misfire on held-out
    episodes — a fixed choice understates linear decodability, which is
    the one direction this instrument must not err in."""
    Xtr = np.asarray(Xtr, dtype=np.float64)
    Xte = np.asarray(Xte, dtype=np.float64)
    ytr = np.asarray(ytr, dtype=np.float64)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Z = (Xtr - mu) / sd
    ym = float(ytr.mean())
    yc = ytr - ym
    eye = np.eye(Z.shape[1])
    grid = [float(alphas)] if np.isscalar(alphas) else list(alphas)
    if len(grid) > 1:
        val = np.random.default_rng(3).random(len(Z)) < 0.2
        fit = ~val
        gram = Z[fit].T @ Z[fit]
        rhs = Z[fit].T @ yc[fit]
        scores = [_r2(yc[val], Z[val] @ np.linalg.solve(gram + a * eye, rhs))
                  for a in grid]
        grid = [grid[int(np.argmax(scores))]]
    w = np.linalg.solve(Z.T @ Z + grid[0] * eye, Z.T @ yc)
    return ((Xte - mu) / sd) @ w + ym


def _ridge_multi(Xtr, Ytr, Xte, alpha: float):
    """Multi-response ridge at fixed alpha (TP2 encoding model: the design
    is low-dimensional, so selection buys nothing and one Gram serves all
    response channels)."""
    Xtr = np.asarray(Xtr, dtype=np.float64)
    Xte = np.asarray(Xte, dtype=np.float64)
    Ytr = np.asarray(Ytr, dtype=np.float64)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Z = (Xtr - mu) / sd
    ym = Ytr.mean(0)
    W = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]),
                        Z.T @ (Ytr - ym))
    return ((Xte - mu) / sd) @ W + ym


def _r2(y, yhat) -> float:
    ss = float(np.sum((y - y.mean()) ** 2))
    return float("nan") if ss == 0 else float(1 - np.sum((y - yhat) ** 2) / ss)


def _auc(y, score) -> float:
    """Rank (Mann-Whitney) AUC with tie-averaged ranks."""
    y = np.asarray(y, dtype=bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    score = np.asarray(score, dtype=float)
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    ranks = sums[inv] / cnt[inv]
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def stage_trunkprobe(lens: PolicyLens, scenario, n_on: int, n_pool: int,
                     max_states: int, alpha_centre: float, out_dir: Path,
                     max_cell_rows: int = 100_000) -> str:
    """TP: does the frozen trunk linearly encode the per-cell relational
    primitives (neighbor exponent diffs, mergeable pairs, occupancy)?

    #E23's caveat left this exact split open: snake_len "dead in the critic
    regression" showed no linear trace in that FEATURE basis, "not proof the
    trunk cannot represent it (the can't-represent vs never-learned-to-use
    split is the D3 probe, not run)". This stage is that probe, pointed at
    the ACTIVATIONS.

    Method: three state pools (the policy's own rollouts / greedy / random —
    off-policy pools test whether the code exists where the policy never
    goes), split by episode parity into train/test; ridge probes per target,
    all metrics held-out. Three probe inputs per cell: the INPUT-PATCH (the
    3x3 window of planes conv layer 1 actually sees — exponent diffs are
    linear in it BY CONSTRUCTION, which is the stage's self-gate: the
    known exact weights must reconstruct every masked row's diff from the
    current-board base-block patch to machine precision (an identity
    check, not a regression — held-out regression on the patch is bounded
    below 1.0 by train-span coverage at 4x4 and so measures the split, not
    the construction; see the in-body comments); mergeable-pair is NOT
    linear
    in it, it is the one-ReLU AND), then L1/L2 single-cell activation
    vectors (+ position one-hot), each read against an architecture-matched
    RANDOM-INIT trunk. Board-level structure targets (mono_lines, snake_len,
    adj_pairs, n_empty, corner_max) ride on the flattened layers.

    Registered before its first full run (ESCALATION.md TP entry, 2026-08-25)
    with the P1 licence thresholds: on the 3x3_20 crown, TRAINED L2 mean
    mergeable AUC < 0.85 OR mean diff R^2 < 0.70 licences a diff-plane
    encoding arm (not built in this case); at-or-above refutes "the
    extractor does not characterize each cell sufficiently" as stated."""
    n = lens.grid_size
    nn_ = n * n
    rng = np.random.default_rng(7)
    alphas = tuple(alpha_centre * f for f in (0.1, 1.0, 10.0, 100.0, 1000.0))

    pools = [("policy", lambda s: lens, [int(s) for s in range(n_on)]),
             ("greedy", lambda s: _GreedyLens(n),
              [5000 + i for i in range(n_pool)]),
             ("random", lambda s: _RandomLens(n, s),
              [6000 + i for i in range(n_pool)])]
    boards, ep_par, pool_id = [], [], []
    pool_sizes = {}
    for pi, (tag, mk, seeds) in enumerate(pools):
        rows = []
        for ei, s in enumerate(seeds):
            for b in rollout(mk(s), scenario, s)["boards"]:
                rows.append((ei % 2, b))
        if len(rows) > max_states:
            keep = sorted(rng.choice(len(rows), size=max_states,
                                     replace=False))
            rows = [rows[i] for i in keep]
        pool_sizes[tag] = len(rows)
        for par, b in rows:
            boards.append(b); ep_par.append(par); pool_id.append(pi)
    S = len(boards)
    ep_par = np.array(ep_par); pool_id = np.array(pool_id)

    obs_np = np.stack([lens.pol._encode(b) for b in boards]
                      ).astype(np.float32)
    P = obs_np.shape[1]

    # targets
    diff = np.empty((S, nn_, 4), np.float32)
    diff_m = np.empty((S, nn_, 4), bool)
    equal = np.empty((S, nn_, 4), np.float32)
    eq_m = np.empty((S, nn_, 4), bool)
    occ = np.empty((S, nn_), np.float32)
    nbrk = np.empty((S, nn_, 4), np.float32)
    kown = np.empty((S, nn_), np.float32)
    bf = np.empty((S, len(FEATURE_NAMES)), np.float32)
    for i, b in enumerate(boards):
        (diff[i], diff_m[i], equal[i], eq_m[i], occ[i],
         nbrk[i], kown[i]) = _cell_targets(b, n)
        f = board_features(b, n)
        bf[i] = [f[k] for k in FEATURE_NAMES]

    # cell-row sample, shared by every per-cell probe input
    all_rows = S * nn_
    row_idx = (np.arange(all_rows) if all_rows <= max_cell_rows else
               np.sort(rng.choice(all_rows, size=max_cell_rows,
                                  replace=False)))
    s_idx, c_idx = row_idx // nn_, row_idx % nn_
    r_idx, cc_idx = c_idx // n, c_idx % n
    pos_oh = np.eye(nn_, dtype=np.float32)[c_idx]
    row_par = ep_par[s_idx]
    row_pool = pool_id[s_idx]

    padded = np.pad(obs_np, ((0, 0), (0, 0), (1, 1), (1, 1)))
    win = np.lib.stride_tricks.sliding_window_view(padded, (3, 3),
                                                   axis=(2, 3))
    patch = win[s_idx, :, r_idx, cc_idx].reshape(len(row_idx), P * 9)
    patch = np.concatenate([patch, pos_oh], axis=1)
    # self-gate patch: the current-board base block ONLY (it leads every
    # onehot-family mode). The exact linear diff rule lives entirely in it;
    # on multi-block modes (_la) the FULL patch adds collinear afterstate
    # copies of the same exponents, ridge spreads the rule's weight across
    # them, and held-out rows where the block correlation shifts pay for it
    # (measured: full-patch 0.9913 vs base-block gate on the same la
    # subject, 2026-08-25) — an estimation artifact, not instrument
    # breakage, so the validity claim is tested where it is exact.
    bp = min(P, nn_ + 1)
    patch_gate = win[s_idx, :bp, r_idx, cc_idx].reshape(len(row_idx), bp * 9)
    patch_gate = np.concatenate([patch_gate, pos_oh], axis=1)

    fe = getattr(lens.pol.model.policy, "features_extractor", None)
    if fe is None:
        fe = lens.pol.model.policy.pi_features_extractor
    fe_rand = _reinit_trunk(fe)

    def cell_mats(trunk):
        acts, feats = _trunk_acts(trunk, obs_np)
        per_cell = {L: np.concatenate(
            [A[s_idx, :, r_idx, cc_idx], pos_oh], axis=1)
            for L, A in acts.items()}
        flat = {L: A.reshape(S, -1) for L, A in acts.items()}
        flat["F"] = feats
        return per_cell, flat

    cell_tr, flat_tr = cell_mats(fe)
    cell_rn, flat_rn = cell_mats(fe_rand)
    layer_names = sorted(cell_tr.keys())

    y_diff = diff.reshape(all_rows, 4)[row_idx]
    m_diff = diff_m.reshape(all_rows, 4)[row_idx]
    y_eq = equal.reshape(all_rows, 4)[row_idx]
    m_eq = eq_m.reshape(all_rows, 4)[row_idx]
    y_occ = occ.reshape(all_rows)[row_idx]
    y_nbrk = nbrk.reshape(all_rows, 4)[row_idx]
    y_nbrocc = (y_nbrk > 0).astype(np.float32)
    k_own_rows = kown.reshape(all_rows)[row_idx]

    def cell_scores(X, sub=None):
        """(diff R^2 x4, equal AUC x4, occ AUC) held-out; sub restricts the
        TEST rows (per-pool breakdown), training always on the full split."""
        te_base = (row_par == 1) if sub is None else ((row_par == 1) & sub)
        out_d, out_e = [], []
        for d in range(4):
            tr = (row_par == 0) & m_diff[:, d]
            te = te_base & m_diff[:, d]
            out_d.append(_r2(y_diff[te, d],
                             _ridge(X[tr], y_diff[tr, d], X[te], alphas)))
            tr = (row_par == 0) & m_eq[:, d]
            te = te_base & m_eq[:, d]
            out_e.append(_auc(y_eq[te, d] > 0.5,
                              _ridge(X[tr], y_eq[tr, d], X[te], alphas)))
        tr, te = row_par == 0, te_base
        o = _auc(y_occ[te] > 0.5, _ridge(X[tr], y_occ[tr], X[te], alphas))
        return np.array(out_d), np.array(out_e), o

    results = {"input": cell_scores(patch)}
    for L in layer_names:
        results[f"{L} rand"] = cell_scores(cell_rn[L])
        results[f"{L} trained"] = cell_scores(cell_tr[L])
    pool_rows = {}
    top = layer_names[-1]
    for pi, (tag, _, _) in enumerate(pools):
        pool_rows[tag] = cell_scores(cell_tr[top], sub=(row_pool == pi))

    # Self-gate by IDENTITY, not regression: the claim "diffs are linear in
    # the base-block patch" is verified with the known exact weights (+j on
    # the centre plane-j column, -j on the direction's neighbour column),
    # reconstructing every masked row's target to machine precision. A
    # regression-based gate was tried first and is bounded ~0.9964 held-out
    # at 4x4 by train-SPAN coverage (novel column COMBINATIONS in held-out
    # episodes — untouched by column-coverage filtering and by alpha
    # selection, 2026-08-25); that is a property of the data split, not of
    # the construction, so it cannot gate instrument validity. Assumes the
    # onehot base block (plane j <=> exponent j); thermo would need nesting.
    _NBR_OFF = {0: 1, 1: 7, 2: 3, 3: 5}   # up/down/left/right, centre = 4
    j_idx = np.arange(bp, dtype=np.float64)
    centre_k = patch_gate[:, 4:bp * 9:9] @ j_idx
    gate_err = 0.0
    for d in range(4):
        nbr_k = patch_gate[:, _NBR_OFF[d]:bp * 9:9] @ j_idx
        pred = centre_k - nbr_k
        m = m_diff[:, d]
        gate_err = max(gate_err, float(np.abs(pred[m] - y_diff[m, d]).max()))
    gate_ok = gate_err < 1e-6

    # board-level structure on flattened layers
    b_tr, b_te = ep_par == 0, ep_par == 1
    b_targets = ["mono_lines", "snake_len", "adj_pairs", "n_empty"]
    b_cols = {k: FEATURE_NAMES.index(k) for k in b_targets + ["corner_max"]}
    board_rows = {}
    flat_in = obs_np.reshape(S, -1)
    for name, X in ([("input", flat_in)]
                    + [(f"{top} rand", flat_rn[top]),
                       (f"{top} trained", flat_tr[top]),
                       ("F rand", flat_rn["F"]),
                       ("F trained", flat_tr["F"])]):
        row = {}
        for k in b_targets:
            y = bf[:, b_cols[k]]
            row[k] = _r2(y[b_te], _ridge(X[b_tr], y[b_tr], X[b_te], alphas))
        y = bf[:, b_cols["corner_max"]]
        row["corner_max"] = _auc(y[b_te] > 0.5,
                                 _ridge(X[b_tr], y[b_tr], X[b_te], alphas))
        board_rows[name] = row

    # ---- TP2 (registered 2026-08-25, the partialling challenge) --------
    # D1 decoding: unique neighbor content beyond Z = [k(own), occ(own),
    # position]; Delta and R^2_Z computed on the SAME row sample so the
    # subtraction is fair. D2 encoding: the user's own direction — all
    # features entered jointly, the relational block scored by its UNIQUE
    # Delta R^2 over the reduced dictionary.
    tp2_rng = np.random.default_rng(11)
    Zn = np.column_stack([k_own_rows, y_occ, pos_oh]).astype(np.float32)
    tp2_inputs = ([("input", patch)]
                  + [(f"{L} {w}", cell_rn[L] if w == "rand" else cell_tr[L])
                     for L in layer_names for w in ("rand", "trained")])

    def tp2_group(y, m, cap=60_000):
        tr = np.flatnonzero((row_par == 0) & m)
        te = np.flatnonzero((row_par == 1) & m)
        if len(tr) > cap:
            tr = np.sort(tp2_rng.choice(tr, cap, replace=False))
        if len(te) > cap:
            te = np.sort(tp2_rng.choice(te, cap, replace=False))
        r2z = _r2(y[te], _ridge(Zn[tr], y[tr], Zn[te], alphas))
        deltas = {}
        for name, X in tp2_inputs:
            zx_tr = np.concatenate([Zn[tr], X[tr]], axis=1)
            zx_te = np.concatenate([Zn[te], X[te]], axis=1)
            deltas[name] = _r2(y[te],
                               _ridge(zx_tr, y[tr], zx_te, alphas)) - r2z
        return r2z, deltas

    tp2 = {"nbr_k": [], "nbr_occ": []}
    for d in range(4):
        onb = m_eq[:, d]
        tp2["nbr_k"].append(
            tp2_group(y_nbrk[:, d], onb & (y_nbrk[:, d] > 0)))
        tp2["nbr_occ"].append(tp2_group(y_nbrocc[:, d], onb))

    d_red = Zn
    d_full = np.concatenate([Zn, y_nbrk, y_nbrocc, y_eq], axis=1)
    enc_rows = {}
    e_tr, e_te = row_par == 0, row_par == 1
    for L in layer_names:
        for w, mats in (("rand", cell_rn), ("trained", cell_tr)):
            Y = mats[L][:, :mats[L].shape[1] - nn_]
            Yte = Y[e_te]
            sst = float(((Yte - Y[e_tr].mean(0)) ** 2).sum())
            r2c = {}
            for tag, D in (("red", d_red), ("full", d_full)):
                pred = _ridge_multi(D[e_tr], Y[e_tr], D[e_te], 1.0)
                r2c[tag] = 1 - float(((Yte - pred) ** 2).sum()) / sst
            enc_rows[f"{L} {w}"] = (r2c["red"], r2c["full"],
                                    r2c["full"] - r2c["red"])

    # report
    top_tr = results[f"{top} trained"]
    eq_mean, diff_mean = float(np.nanmean(top_tr[1])), float(np.nanmean(top_tr[0]))
    licence = eq_mean < 0.85 or diff_mean < 0.70
    lines = [
        "# Trunk representation probe (TP — #E23's open \"D3\" instrument)",
        f"subject: {out_dir.parent.name}",
        f"pools: policy {n_on} eps ({pool_sizes['policy']} states) / greedy "
        f"{n_pool} ({pool_sizes['greedy']}) / random {n_pool} "
        f"({pool_sizes['random']}); split = episode parity; "
        f"ridge alpha val-selected from {min(alphas):g}..{max(alphas):g}; "
        f"{len(row_idx)} cell rows; all metrics held-out",
        f"self-gate: base-block patch reconstructs the diffs by identity, "
        f"max |err| = {gate_err:.2e} "
        f"[{'PASS' if gate_ok else 'FAIL — INSTRUMENT BROKEN, IGNORE BELOW'}]\n",
        "## Per-cell primitives (probe input = single-cell activation + "
        "position one-hot; `input` = the 3x3 plane patch layer 1 sees)",
        "| probe input | diff R^2 mean | diff R^2 worst | mergeable AUC mean "
        "| mergeable AUC worst | occupancy AUC |", "|---|---|---|---|---|---|"]
    for name in ["input"] + [f"{L} {w}" for L in layer_names
                             for w in ("rand", "trained")]:
        d, e, o = results[name]
        lines.append(f"| {name} | {np.nanmean(d):.3f} | {np.nanmin(d):.3f} "
                     f"| {np.nanmean(e):.3f} | {np.nanmin(e):.3f} | {o:.3f} |")
    lines += ["\n## Board-level structure (flattened layer, held-out)",
              "| probe input | " + " | ".join(b_targets) + " | corner_max AUC |",
              "|---|" + "---|" * (len(b_targets) + 1)]
    for name, row in board_rows.items():
        lines.append("| " + name + " | "
                     + " | ".join(f"{row[k]:.3f}" for k in b_targets)
                     + f" | {row['corner_max']:.3f} |")
    lines += [f"\n## Per-pool ({top} trained)",
              "| pool | diff R^2 mean | mergeable AUC mean |", "|---|---|---|"]
    for tag, (d, e, _) in pool_rows.items():
        lines.append(f"| {tag} | {np.nanmean(d):.3f} | {np.nanmean(e):.3f} |")
    lines += ["\n## TP2 — partialled unique neighbor content "
              "(registered 2026-08-25; interpretive, no new licence)",
              "D1 decoding: target k(neighbor_d) | Z = [k(own), occ(own), "
              "position]; Delta = R^2_[Z,h] - R^2_Z, shared rows.",
              f"R^2_Z alone (4-dir means): k-neighbor "
              f"{np.mean([g[0] for g in tp2['nbr_k']]):.3f} · neighbor-occ "
              f"{np.mean([g[0] for g in tp2['nbr_occ']]):.3f}",
              "| probe input | Delta R^2 k-nbr | partial R^2 k-nbr "
              "| Delta R^2 nbr-occ |", "|---|---|---|---|"]
    for name, _ in tp2_inputs:
        dk = [g[1][name] for g in tp2["nbr_k"]]
        pk = [g[1][name] / max(1e-9, 1 - g[0]) for g in tp2["nbr_k"]]
        do = [g[1][name] for g in tp2["nbr_occ"]]
        lines.append(f"| {name} | {np.mean(dk):.3f} | {np.mean(pk):.3f} "
                     f"| {np.mean(do):.3f} |")
    lines += ["\nD2 encoding: h(cell) ~ all features jointly; unique "
              "Delta of the relational block [k-nbr x4, nbr-occ x4, "
              "equal x4] over [position, k(own), occ(own)]; alpha 1.0.",
              "| layer | R^2 reduced | R^2 full | Delta relational |",
              "|---|---|---|---|"]
    for name, (r_red, r_full, dl) in enc_rows.items():
        lines.append(f"| {name} | {r_red:.3f} | {r_full:.3f} | {dl:.3f} |")
    lines += ["\n## Registered thresholds (TP entry, 2026-08-25)",
              f"- {top} trained mergeable AUC mean = {eq_mean:.3f} "
              f"(licence threshold: < 0.85)",
              f"- {top} trained diff R^2 mean = {diff_mean:.3f} "
              f"(licence threshold: < 0.70)",
              f"- P1 (diff-plane arm) licence: "
              f"{'FIRED' if licence else 'NOT FIRED'}"]
    report = "\n".join(lines) + "\n"
    (out_dir / "trunkprobe.md").write_text(report)
    if not gate_ok:
        raise SystemExit(f"trunkprobe self-gate FAILED: identity "
                         f"reconstruction max |err| = {gate_err:.2e}")
    return report


# ---------------------------------------------------------------------------
# stage: counterfactual (rung 3, CRN branches)
# ---------------------------------------------------------------------------

def stage_counterfactual(lens: PolicyLens, scenario, n_states: int,
                         out_dir: Path, n_salts: int = 6,
                         critic: str = "state", probe_seed: int = 1) -> str:
    """Per-action score-to-go from sampled states.

    A single CRN branch conflates decision quality with bifurcated luck (one
    different move changes the empties, hence every later spawn cell; episode
    std is ~800 pts). So each branch's future is re-randomized over
    ``n_salts`` seed salts — the past stays frozen, only draws after the
    branch point change — and Q(s,a) is the salt-mean. The raw single-branch
    numbers are reported alongside as the luck-included view."""
    import dataclasses

    rng = np.random.default_rng(probe_seed)
    regrets_q, regrets_raw, chose_best_q, rank_agree, rows = [], [], 0, [], 0
    ep_scores = []  # the regret-as-share denominator, measured not hardcoded
    seeds = rng.choice(2000, size=max(4, n_states // 6), replace=False) + 100_000

    def branch_outcome(st, a, salt):
        sb = dataclasses.replace(st, board=list(st.board), seed_salt=salt)
        s2, info = advance(scenario, sb, a)
        tot = info["gained"]
        while True:
            va2 = valid_actions(s2)
            if not any(va2) or s2.period >= scenario.T_cap:
                break
            s2, i2 = advance(scenario, s2, lens.act(list(s2.board)))
            tot += i2["gained"]
        return tot, list(s2.board)

    for seed in seeds:
        ep = rollout(lens, scenario, int(seed))
        if len(ep["boards"]) < 10:
            continue
        ep_scores.append(float(ep["gains"].sum()))
        for t in sorted(rng.choice(len(ep["boards"]) - 1,
                                   size=min(6, len(ep["boards"]) - 1),
                                   replace=False)):
            st = open_episode(scenario, init_state(scenario, int(seed))[0])
            for k in range(t):
                st, _ = advance(scenario, st, ep["actions"][k])
            va = valid_actions(st)
            if sum(va) < 2:
                continue
            salts = [st.seed_salt] + [90_000 + i for i in range(1, n_salts)]
            acts = [a for a, ok in enumerate(va) if ok]
            q_true = {a: np.mean([branch_outcome(st, a, s)[0] for s in salts])
                      for a in acts}
            raw = {a: branch_outcome(st, a, st.seed_salt)[0] for a in acts}
            qhat = []
            for a in acts:
                if critic == "afterstate":
                    # rung-2a models: the critic is defined on the post-merge
                    # pre-spawn board, so the branch score gained + V(s^a) is
                    # EXACT given V — no spawn sample stands in the way
                    after, gained, _ = apply_move(list(st.board), a)
                    qhat.append(gained + lens.values([after])[0])
                else:
                    s2, info = advance(
                        scenario,
                        dataclasses.replace(st, board=list(st.board)), a)
                    qhat.append(info["gained"]
                                + lens.values([list(s2.board)])[0])
            a_act = ep["actions"][t]
            best_q = max(q_true.values())
            regrets_q.append(best_q - q_true[a_act])
            regrets_raw.append(max(raw.values()) - raw[a_act])
            chose_best_q += q_true[a_act] >= best_q - 1e-9
            rank_agree.append(_spearman(qhat, [q_true[a] for a in acts]))
            rows += 1
            if rows >= n_states:
                break
        if rows >= n_states:
            break
    rq, rr = np.array(regrets_q), np.array(regrets_raw)
    se = rq.std() / max(np.sqrt(rows), 1)
    mean_ep = float(np.mean(ep_scores)) if ep_scores else float("nan")
    lines = ["# CRN counterfactual branches: Q(s,a) with re-salted futures",
             f"subject: {out_dir.parent.name}",
             f"provisional: {rows} states x {n_salts} future salts per branch"
             f" (probe_seed={probe_seed});",
             "past frozen, futures re-randomized; continuation = the policy itself\n",
             "## Decision quality (salt-averaged Q — luck integrated out)",
             f"- actor's move is Q-optimal: {chose_best_q}/{rows} "
             f"({chose_best_q / rows:.1%})",
             f"- mean Q-regret of the actor's move: {rq.mean():.1f} ± {se:.1f} pts "
             f"(median {np.median(rq):.1f}, p90 {np.percentile(rq, 90):.1f})",
             f"- Q-regret as share of mean episode score: "
             f"{100 * rq.mean() / mean_ep:.1f}%  (mean episode = {mean_ep:.0f} pts "
             f"over the {len(ep_scores)} sampled rollouts)",
             f"- critic ranks branches correctly (spearman of "
             f"[gained + V({'s^a' if critic == 'afterstate' else 's-prime'})] "
             f"vs Q): {np.nanmean(rank_agree):.3f}  [critic form: {critic}]",
             "\n## Luck-included view (single shared-salt branch, for contrast)",
             f"- mean raw-branch regret: {rr.mean():.1f} pts "
             f"(p90 {np.percentile(rr, 90):.1f}) — the excess over the "
             "Q-regret above is bifurcated luck, not decision error"]
    report = "\n".join(lines) + "\n"
    (out_dir / "counterfactual.md").write_text(report)
    return report


# ---------------------------------------------------------------------------
# stage: symmetry (A9 rung 1 — orbit probes against the D4 engine fact)
# ---------------------------------------------------------------------------

def stage_symmetry(lens: PolicyLens, scenario, n_seeds: int, n_states: int,
                   out_dir: Path, n_salts: int = 3) -> str:
    """How the crown sits relative to the certified D4 symmetry (test 39).

    Part A — actor/critic orbit statistics over on-policy states. For g != e,
    g·s is OFF the crown's visited distribution (it never sees boards homed
    away from its corner), so actor orbit-DISAGREEMENT measures the
    commitment's footprint, not error: the optimal set is D4-closed and a
    committed corner policy is a legitimate broken-symmetry member of it.

    Part B — the sharp question: does the critic KNOW the commitment?
    V^pi(g·s) genuinely varies over the orbit for a committed pi (a board
    homed to the wrong corner really is worth less TO THIS POLICY), so the
    orbit spread of V-hat is not itself an error. The error question is
    whether V-hat(g·s) TRACKS the realized CRN return from g·s — a critic
    reading only D4-invariant structure (the regression basis of the pilot
    is invariant) would be flat across the orbit while the truth is not.
    """
    elems = d4_transforms(lens.grid_size)
    non_id = [(nm, p, mp) for nm, p, mp in elems if nm != "identity"]

    # -- Part A: orbit statistics over on-policy states ----------------------
    boards = []
    for seed in range(n_seeds):
        boards += rollout(lens, scenario, seed)["boards"]
    agree = {nm: 0 for nm, _, _ in non_id}
    tv = {nm: 0.0 for nm, _, _ in non_id}
    v_spread, v_all = [], []
    chunk = 256
    for lo in range(0, len(boards), chunk):
        blk = boards[lo:lo + chunk]
        p_e = lens.probs_batch(blk)
        v_orbit = np.empty((len(blk), 8))
        v_orbit[:, 0] = lens.values(blk)
        for gi, (nm, perm, mp) in enumerate(non_id):
            tblk = [[b[p] for p in perm] for b in blk]
            p_g = lens.probs_batch(tblk)
            v_orbit[:, gi + 1] = lens.values(tblk)
            # equivariance would give p_g[:, mp[m]] == p_e[:, m]
            agree[nm] += int(np.sum(np.argmax(p_g, axis=1)
                                    == np.array(mp)[np.argmax(p_e, axis=1)]))
            conj = np.empty_like(p_e)
            for m in range(4):
                conj[:, mp[m]] = p_e[:, m]
            tv[nm] += float(np.sum(0.5 * np.abs(p_g - conj).sum(axis=1)))
        v_spread.append(v_orbit.max(axis=1) - v_orbit.min(axis=1))
        v_all.append(v_orbit[:, 0])
    n_st = len(boards)
    v_spread = np.concatenate(v_spread)
    v_mean = float(np.concatenate(v_all).mean())

    # -- Part B: V-hat vs realized CRN return, over orbits of held states ----
    import dataclasses

    def continuation(st, salt):
        s2 = dataclasses.replace(st, board=list(st.board), seed_salt=salt)
        tot = 0.0
        while any(valid_actions(s2)) and s2.period < scenario.T_cap:
            s2, info = advance(scenario, s2, lens.act(list(s2.board)))
            tot += info["gained"]
        return tot

    rng = np.random.default_rng(9)
    seeds = rng.choice(2000, size=n_states, replace=False) + 200_000
    spread_r, spread_v, corr_rv, cost_wrong = [], [], [], []
    used = 0
    for seed in seeds:
        ep = rollout(lens, scenario, int(seed))
        if len(ep["boards"]) < 12:
            continue
        t = int(rng.integers(5, len(ep["boards"]) - 5))
        st = open_episode(scenario, init_state(scenario, int(seed))[0])
        for k in range(t):
            st, _ = advance(scenario, st, ep["actions"][k])
        salts = [90_000 + i for i in range(n_salts)]
        r_orbit, v_orbit = [], []
        for nm, perm, _ in elems:
            tb = [st.board[p] for p in perm]
            st_g = dataclasses.replace(st, board=tb)
            r_orbit.append(np.mean([continuation(st_g, s) for s in salts]))
            v_orbit.append(lens.values([tb])[0])
        r_o, v_o = np.array(r_orbit), np.array(v_orbit)
        spread_r.append(r_o.max() - r_o.min())
        spread_v.append(v_o.max() - v_o.min())
        corr_rv.append(_pearson(v_o, r_o))
        cost_wrong.append(r_o[0] - r_o[1:].mean())
        used += 1

    lines = [
        "# Symmetry probes (A9 rung 1) — the policy vs the certified D4 engine fact",
        f"subject: {out_dir.parent.name}",
        f"provisional: {n_seeds} rollouts / {n_st} states (part A); "
        f"{used} orbit states x 8 transforms x {n_salts} CRN salts (part B)\n",
        "## Part A — the commitment's footprint (NOT an error measure)",
        f"- actor argmax agreement with its own conjugate, per transform "
        f"(equivariant policy = 100%):",
    ]
    for nm, _, _ in non_id:
        lines.append(f"    {nm:>14}: {agree[nm] / n_st:6.1%}   "
                     f"mean TV dist {tv[nm] / n_st:.3f}")
    lines += [
        f"- mean over non-identity transforms: "
        f"{sum(agree.values()) / (7 * n_st):.1%} "
        f"(chance under masks ~ uniform-over-valid)",
        f"- critic orbit spread on visited states: mean {v_spread.mean():.0f} pts "
        f"(p90 {np.percentile(v_spread, 90):.0f}) against mean V {v_mean:.0f}\n",
        "## Part B — does the critic KNOW the commitment? (V-hat vs CRN truth "
        "across each orbit)",
        f"- realized return spread across an orbit: mean "
        f"{np.mean(spread_r):.0f} pts — the commitment is genuinely priced "
        f"by the world",
        f"- critic's spread across the same orbits: mean {np.mean(spread_v):.0f} pts",
        f"- ratio V-spread / R-spread: {np.mean(spread_v) / np.mean(spread_r):.2f} "
        f"(1 = fully aware, 0 = orientation-blind)",
        f"- mean per-orbit pearson(V-hat, R): {np.nanmean(corr_rv):.3f}",
        f"- orientation cost to the crown: R(identity) - mean R(transformed) = "
        f"{np.mean(cost_wrong):.0f} pts",
    ]
    report = "\n".join(lines) + "\n"
    (out_dir / "symmetry.md").write_text(report)
    return report


# ---------------------------------------------------------------------------
# stage: frameavg (A9 rung 2 — pre-registered eval of the symmetrized crown)
# ---------------------------------------------------------------------------

class _FrameAveragedCrown:
    """The crown symmetrized at eval time: mean conjugated action probabilities
    over the 8 transforms, deterministic argmax over the legal moves.

    Exactly equivariant by construction. Pre-registered prediction
    (ESCALATION.md A9): this HURTS — the eight conjugated policies each home a
    different corner, and averaging dissolves the commitment.
    """

    def __init__(self, lens: PolicyLens):
        self.lens = lens
        self.elems = d4_transforms(lens.grid_size)

    def act(self, board: list[int], mask) -> int:
        orbit = [[board[p] for p in perm] for _, perm, _ in self.elems]
        probs = self.lens.probs_batch(orbit)
        avg = np.zeros(4)
        for gi, (_, _, mp) in enumerate(self.elems):
            for m in range(4):
                avg[m] += probs[gi, mp[m]]
        avg = np.where(np.asarray(mask, dtype=bool), avg, -np.inf)
        return int(np.argmax(avg))


def stage_frameavg(lens: PolicyLens, scenario_name: str, n_seeds: int,
                   out_dir: Path) -> None:
    from game2048_benchmark_common import evaluate

    ns = argparse.Namespace(scenario=scenario_name, n_seeds=n_seeds,
                            outdir="results")
    evaluate("frameavg_crown", lambda env: _FrameAveragedCrown(lens), ns,
             out_path=out_dir / f"frameavg_eval_{scenario_name}.tsv",
             note=" | mean conjugated probs over D4, deterministic argmax")


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["replay", "structure", "probes",
                                      "counterfactual", "symmetry",
                                      "frameavg", "trunkprobe"])
    ap.add_argument("--model-path", required=True)
    ap.add_argument("-s", "--scenario", default="3x3_20",
                    choices=list(SCENARIOS.keys()))
    ap.add_argument("--obs-mode", dest="obs_mode", default="onehot",
                    choices=list(OBSERVATION_MODES),
                    help="observation mode the model was TRAINED with (read "
                         "it from the run dir's obs... field). It must match, "
                         "or the policy wrapper raises on the obs shape")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--n-seeds", type=int, default=64)
    ap.add_argument("--n-states", type=int, default=60)
    ap.add_argument("--probe-seed", dest="probe_seed", type=int, default=1,
                    help="counterfactual: seed for the state sample (episode "
                         "seeds + within-episode timesteps). Default 1 is the "
                         "value hardcoded before 2026-08-05 and reproduces "
                         "every previously recorded reading; vary it to put "
                         "an error bar on the branch-ranking gate. Note the "
                         "generic --seed does NOT reach this stage.")
    ap.add_argument("--n-orbit-states", type=int, default=24)
    ap.add_argument("--max-frames", type=int, default=240,
                    help="replay: cap on GIF frames (the episode is strided "
                         "to fit; the .txt log always keeps every step)")
    ap.add_argument("--critic", default="state",
                    choices=["state"],
                    help="counterfactual: how the model's critic scores a "
                         "branch. 'afterstate' scores it as "
                         "EXACT gained + V(post-merge board); 'state' uses "
                         "gained + V(post-spawn board), one spawn sample. "
                         "Other stages read V(board) as-is either way — for "
                         "an afterstate model that means 'board scored as an "
                         "afterstate', one event earlier than a state critic")
    ap.add_argument("--with-greedy", action="store_true",
                    help="structure: append a myopic-baseline row through "
                         "the same instrument (anchors the mono scale)")
    ap.add_argument("--n-pool-seeds", dest="n_pool_seeds", type=int,
                    default=32,
                    help="trunkprobe: episodes per OFF-policy pool (greedy "
                         "and random each); --n-seeds sets the on-policy "
                         "count as in the other stages")
    ap.add_argument("--max-states", dest="max_states", type=int, default=8000,
                    help="trunkprobe: per-pool state cap (uniform seeded "
                         "subsample above it)")
    ap.add_argument("--probe-alpha", dest="probe_alpha", type=float,
                    default=1.0,
                    help="trunkprobe: ridge strength on standardized "
                         "columns; held-out metrics guard overfit, this "
                         "only conditions the solve")
    args = ap.parse_args(argv)

    sc = SCENARIOS[args.scenario]
    sc = sc(0) if callable(sc) else sc
    lens = PolicyLens(args.model_path, grid_size=sc.grid_size,
                      observation_mode=args.obs_mode)
    _state_at.lens = lens
    out_dir = Path(args.model_path).resolve().parent / "interpret"
    out_dir.mkdir(exist_ok=True)

    if args.stage == "replay":
        stage_replay(lens, sc, args.seed, out_dir, args.max_frames)
    elif args.stage == "structure":
        print(stage_structure(lens, sc, args.n_seeds, out_dir,
                              args.with_greedy))
    elif args.stage == "probes":
        print(stage_probes(lens, sc, args.n_seeds, out_dir))
    elif args.stage == "symmetry":
        print(stage_symmetry(lens, sc, args.n_seeds, args.n_orbit_states,
                             out_dir))
    elif args.stage == "frameavg":
        stage_frameavg(lens, args.scenario, args.n_seeds, out_dir)
    elif args.stage == "trunkprobe":
        print(stage_trunkprobe(lens, sc, args.n_seeds, args.n_pool_seeds,
                               args.max_states, args.probe_alpha, out_dir))
    else:
        print(stage_counterfactual(lens, sc, args.n_states, out_dir,
                                   critic=args.critic,
                                   probe_seed=args.probe_seed))


if __name__ == "__main__":
    main()

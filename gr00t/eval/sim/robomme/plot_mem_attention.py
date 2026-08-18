#!/usr/bin/env python3
"""Plot the memory-attention attribution written by GR00T_MEM_ATTR=1.

Companion to aggregate_mem_attention.py: that one reduces the per-call JSONL to numbers,
this one draws it, because the shape over a rollout is what the numbers hide -- whether
the action head locks onto one pooled step, drifts across the window as the episode
progresses, or reads nothing but the live frame from beginning to end.

Two figures per task:

    mem_attn/plots/<episode>.png   attention share of every block of `mem_seq` over the
                                   calls of that episode (blocks x calls), with the
                                   share the action tokens put on memory at all below it
    mem_attention_scores.png       one row per episode: that episode's mean share per
                                   block, so ten rollouts fit in one picture

Both are single-hue heatmaps: the quantity is a magnitude (a softmax share in [0,1]),
so it gets a light->dark ramp, not a rainbow. The dotted marks on the colorbar are the
uniform baseline 1/K -- the level every block would sit at if the memory transformer
spread its attention evenly -- which is the only reference that makes a raw share
readable. The current block is structurally above it (the rollout's residual term keeps
mass on the querying token's own block), so compare pooled blocks to the baseline and to
each other, not to the current block.

Usage:
    python gr00t/eval/sim/robomme/plot_mem_attention.py <run_dir>          # every task
    python gr00t/eval/sim/robomme/plot_mem_attention.py <run_dir>/<TASK>   # one task
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")  # eval hosts are headless
import matplotlib.pyplot as plt
import numpy as np

TASKS = [
    "BinFill", "PickXtimes", "SwingXtimes", "StopCube",
    "VideoUnmask", "VideoUnmaskSwap", "ButtonUnmask", "ButtonUnmaskSwap",
    "PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder",
    "MoveCube", "InsertPeg", "PatternLock", "RouteStick",
]
_KIND_TAG = {"pool": "P", "recent": "R", "current": "C", "window": "W"}

# Recessive frame, ink for text, one sequential ramp for the data.
INK, MUTED, GRID = "#1f2933", "#5b6470", "#d8dde3"
RAMP = "Blues"


def _episodes(task_dir: Path) -> dict[str, list[dict]]:
    """{episode -> its calls, in order}, one entry per mem_attn/<session_id>.jsonl."""
    mem = task_dir / "mem_attn"
    if not mem.is_dir():
        return {}
    out = {}
    for p in sorted(mem.glob("*.jsonl")):
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        if rows:
            out[p.stem] = sorted(rows, key=lambda r: r.get("call_index", 0))
    return out


def _matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(scores[K, n_calls], mass[n_calls], block labels).

    Rows sharing a call_index are averaged: one JSONL can hold several batch samples per
    call, and they are the same call of the same episode.
    """
    by_call: dict[int, list[dict]] = {}
    for r in rows:
        by_call.setdefault(r.get("call_index", len(by_call)), []).append(r)
    calls = sorted(by_call)
    K = rows[0]["window"]
    scores = np.zeros((K, len(calls)))
    mass = np.zeros(len(calls))
    for j, c in enumerate(calls):
        group = by_call[c]
        for s in group[0]["steps"]:
            scores[s["block_index"], j] = np.mean(
                [g["steps"][s["block_index"]]["score"] for g in group]
            )
        mass[j] = np.mean([g["memory_attention_mass"] for g in group])
    labels = [
        f"b{s['block_index']}{_KIND_TAG.get(s.get('slot_kind', 'pool'), '?')}"
        for s in rows[0]["steps"]
    ]
    return scores, mass, labels


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)


def _kind_breaks(labels: list[str]) -> list[float]:
    """Row boundaries between pooled / recent / current, for a separator line."""
    return [i - 0.5 for i in range(1, len(labels)) if labels[i][-1] != labels[i - 1][-1]]


def _colorbar(fig, im, ax, K: int, label: str = "attention share"):
    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.025)
    cb.set_label(label, color=MUTED, fontsize=8)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    cb.outline.set_visible(False)
    # The uniform level, the only reference that makes a raw share readable.
    cb.ax.axhline(1.0 / K, color=INK, lw=1.2, ls=":")
    return cb


def plot_episode(episode: str, rows: list[dict], out: Path) -> None:
    """Attention share of every HISTORY block over the calls of one episode.

    The current block is deliberately not a row of the heatmap: its share is ~1/2**L
    from the rollout's residual term alone, several times any pooled block, and putting
    it on the same color scale flattens everything else to white. It gets a line below
    instead, where its level is readable and its constancy is the point.
    """
    scores, mass, labels = _matrix(rows)
    hist, cur = scores[:-1], scores[-1]
    K, n = scores.shape
    H = K - 1
    fig, (ax, axl) = plt.subplots(
        2, 1, figsize=(max(6.0, min(14.0, 2.2 + 0.09 * n)), 1.8 + 0.26 * H),
        sharex=True, height_ratios=[H, max(3.0, H * 0.30)],
        constrained_layout=True,
    )

    im = ax.imshow(hist, aspect="auto", origin="upper", cmap=RAMP,
                   vmin=0.0, vmax=max(hist.max(), 1.5 / K),
                   extent=(-0.5, n - 0.5, H - 0.5, -0.5), interpolation="nearest")
    ax.set_yticks(range(H), labels[:-1], fontsize=7)
    ax.set_ylabel("mem_seq block (history)", color=MUTED, fontsize=8)
    for y in _kind_breaks(labels[:-1]):  # pooled | recent
        ax.axhline(y, color="white", lw=1.6)
    _style(ax)
    _colorbar(fig, im, ax, K)

    dark, light = plt.get_cmap(RAMP)(0.85), plt.get_cmap(RAMP)(0.5)
    axl.plot(range(n), cur, color=dark, lw=2.0, label="current block (of window)")
    axl.plot(range(n), mass, color=light, lw=2.0, ls="--", label="memory (of all cross-attn)")
    axl.set_ylim(0, max(float(max(cur.max(), mass.max())) * 1.25, 1e-3))
    axl.set_ylabel("share", color=MUTED, fontsize=8)
    axl.set_xlabel("policy call", color=MUTED, fontsize=8)
    axl.grid(axis="y", color=GRID, lw=0.6, alpha=0.6)
    axl.set_axisbelow(True)
    leg = axl.legend(frameon=False, fontsize=7, loc="upper left", ncols=2,
                     labelcolor=MUTED, handlelength=1.6)
    leg.set_zorder(5)
    _style(axl)

    ax.set_title(
        f"{episode}\n{n} calls · K={K} · uniform baseline {100 / K:.1f}% "
        f"(dotted on the bar) · current block {100 * cur.mean():.0f}% mean, shown below "
        f"· P pooled · R recent",
        color=INK, fontsize=9, loc="left", pad=8,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_task(task: str, per_ep: dict[str, list[dict]], out: Path) -> None:
    """One row per episode: its mean share per block over its own calls.

    Same split as the episode figure -- history blocks in the heatmap, the current block
    printed in a column beside it so it cannot swallow the scale.
    """
    eps = list(per_ep)
    mat = np.stack([_matrix(rows)[0].mean(axis=1) for rows in per_ep.values()])  # (E, K)
    labels = _matrix(next(iter(per_ep.values())))[2]
    E, K = mat.shape
    hist, cur = mat[:, :-1], mat[:, -1]
    H = K - 1

    fig, ax = plt.subplots(figsize=(max(6.0, 2.4 + 0.5 * H), 1.8 + 0.34 * E),
                           constrained_layout=True)
    im = ax.imshow(hist, aspect="auto", cmap=RAMP, vmin=0.0,
                   vmax=max(hist.max(), 1.5 / K), interpolation="nearest")
    ax.set_xticks(range(H), labels[:-1], fontsize=7)
    ax.set_yticks(range(E), [e[:28] for e in eps], fontsize=7)
    for x in _kind_breaks(labels[:-1]):
        ax.axvline(x, color="white", lw=1.6)
    if E * H <= 320:  # readable at this density; above it the colors carry it alone
        hi = hist.max()
        for i in range(E):
            for j in range(H):
                ax.text(j, i, f"{100 * hist[i, j]:.0f}", ha="center", va="center",
                        fontsize=6, color="white" if hist[i, j] > 0.62 * hi else INK)
    # The current block as text, outside the mapped area: a value, not a color.
    ax.text(1.004, 1.006, labels[-1], transform=ax.transAxes, fontsize=7,
            color=MUTED, ha="left", va="bottom")
    for i, v in enumerate(cur):
        ax.text(H - 0.3, i, f"{100 * v:.0f}", fontsize=6.5, color=MUTED,
                ha="left", va="center")
    _style(ax)
    ax.tick_params(length=0)
    _colorbar(fig, im, ax, K, "mean attention share")
    ax.set_title(
        f"{task} · mean attention share per block, per episode\n"
        f"{E} episodes · K={K} · uniform baseline {100 / K:.1f}% · cells are % "
        f"· {labels[-1]} printed at right, off the color scale",
        color=INK, fontsize=9, loc="left", pad=16,
    )
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(target: str) -> None:
    root = Path(target)
    tasks = (
        {root.name: root}
        if (root / "mem_attn").is_dir()
        else {t: root / t for t in TASKS if (root / t / "mem_attn").is_dir()}
    )
    if not tasks:
        print(f"[!] no mem_attn/*.jsonl under {root} — was GR00T_MEM_ATTR=1 set?")
        sys.exit(1)

    for task, task_dir in tasks.items():
        per_ep = _episodes(task_dir)
        if not per_ep:
            continue
        for ep, rows in per_ep.items():
            plot_episode(ep, rows, task_dir / "mem_attn" / "plots" / f"{ep}.png")
        out = task_dir / "mem_attention_scores.png"
        plot_task(task, per_ep, out)
        print(f"[i] {task}: {len(per_ep)} episode plots + {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])

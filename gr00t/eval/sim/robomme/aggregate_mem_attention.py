#!/usr/bin/env python3
"""Summarize the memory-attention attribution written by GR00T_MEM_ATTR=1.

Reads `<run_dir>/<TASK>/mem_attn/memory_attention.jsonl` (one JSON line per policy call,
written by Gr00tPolicy via gr00t/model/modules/memory_attribution.py) and reports, per
task, WHICH pooled step the action head read:

Every slot of the window competes, tagged by what it is: P = pooled (won the admission
contest), R = reserved-recent (handed the last observations unconditionally), C = the
live observation. That separation is the point -- attention landing on R or C is not
evidence the pool selected well.

    mass        mean probability the action tokens put on the memory tokens at all.
                Near zero means memory is barely being used (or the key-moment gate
                zeroed it), and the rest of the row says little.
    won         in how many EPISODES each slot led, oldest slot first -- an episode
                votes once, for the block with its highest mean score, so the counts sum
                to the episode count and a long rollout cannot outvote several short
                ones.
    mean        each slot's mean score over ALL calls, not just the ones it won -- a
                slot can matter steadily without ever taking the argmax, which `won`
                alone hides.
    winner      the same argmax broken down by slot kind. Mostly C means the window adds
                nothing over the frame the backbone just encoded; mostly R means a short
                FIFO would do; P is the only share that credits the pool.
    age         how far back the winning step was captured, in policy calls, over the
                history winners (C is always 0 and would only dilute it). This is the
                quantity a long-horizon memory is supposed to make large.
    padding     share of calls whose winner was a warm-up duplicate rather than a real
                pooled memory -- high means the pool is still filling (or restarting)
                for much of the rollout, so the ranking above is not about memory.

Usage:
    python gr00t/eval/sim/robomme/aggregate_mem_attention.py <run_dir>          # all tasks
    python gr00t/eval/sim/robomme/aggregate_mem_attention.py <run_dir>/<TASK>   # one task
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import statistics
import sys

SUITES = {
    "Counting": ["BinFill", "PickXtimes", "SwingXtimes", "StopCube"],
    "Permanence": ["VideoUnmask", "VideoUnmaskSwap", "ButtonUnmask", "ButtonUnmaskSwap"],
    "Reference": ["PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder"],
    "Imitation": ["MoveCube", "InsertPeg", "PatternLock", "RouteStick"],
}
TASKS = [t for tasks in SUITES.values() for t in tasks]


def _rows(jsonl: Path) -> list[dict]:
    out = []
    with open(jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _winner(row: dict) -> dict | None:
    """Highest-scoring block of the window, live observation included.

    Every slot competes: which step the memory transformer makes the action read is the
    question, and "the current one" is a real answer -- it means the K-block window is
    adding nothing over the frame the backbone just encoded.
    """
    return max(row["steps"], key=lambda s: s["score"]) if row["steps"] else None


def _age(row: dict, win: dict) -> int | None:
    """How many policy calls back the winning block was captured.

    zoo step ids are the pool's own step counter (the zoo tick at inference, one per
    call), so the difference is in calls. The rolling window labels blocks with relative
    offsets already, so its age is just -offset.
    """
    sid, qid = win["step_id"], row["query_step_id"]
    if sid is None:
        return None
    if row.get("memory_mode") == "zoo":
        return None if qid is None else int(qid) - int(sid)
    return -int(sid)


def summarize(rows: list[dict]) -> dict:
    wins = [(r, _winner(r)) for r in rows]
    wins = [(r, w) for r, w in wins if w is not None]
    if not wins:
        return {}
    window = rows[0]["window"]
    # Ages of the winners that are actual history: the live block is always age 0 and
    # would drag the mean toward it without saying anything about the memory.
    ages = [_age(r, w) for r, w in wins if not w["is_current"]]
    ages = [a for a in ages if a is not None]
    kinds = [w.get("slot_kind", "pool") for _, w in wins]
    # Episodes won: each episode is scored by its own mean per block and contributes one
    # vote, to the block that leads it. Call-level counting lets a long rollout outvote
    # nine short ones; this asks instead "in how many episodes was this block the one the
    # action head read most", which is the question a per-episode file layout implies.
    by_ep: dict[str, dict[int, list[float]]] = {}
    for r in rows:
        ep = r.get("session_id") or r.get("episode_id")
        slot = by_ep.setdefault(ep, {})
        for st in r["steps"]:
            slot.setdefault(st["block_index"], []).append(st["score"])
    ep_wins = Counter(
        max(slot, key=lambda i: statistics.fmean(slot[i])) for slot in by_ep.values()
    )
    # Mean score of each slot over ALL calls (not just the ones it won): a slot can
    # matter steadily without ever taking the argmax.
    per_slot: dict[int, list[float]] = {}
    for r in rows:
        for s in r["steps"]:
            per_slot.setdefault(s["block_index"], []).append(s["score"])
    return {
        "calls": len(rows),
        "episodes": len({r.get("session_id") or r.get("episode_id") for r in rows}),
        "mass": statistics.fmean(r["memory_attention_mass"] for r in rows),
        "slots": Counter(w["block_index"] for _, w in wins),
        "ep_wins": ep_wins,
        "n_ep": len(by_ep),
        "slot_mean": {i: statistics.fmean(v) for i, v in per_slot.items()},
        "kinds": Counter(kinds),
        "kind_of": {s["block_index"]: s.get("slot_kind", "pool") for s in rows[0]["steps"]},
        "window": window,
        "age_mean": statistics.fmean(ages) if ages else float("nan"),
        "age_p50": statistics.median(ages) if ages else float("nan"),
        "age_max": max(ages) if ages else float("nan"),
        "age_n": len(ages),
        "padding": statistics.fmean(float(w["is_padding"]) for _, w in wins),
        "source": rows[0].get("source", "?"),
    }


_KIND_TAG = {"pool": "P", "recent": "R", "current": "C", "window": "W"}


def _fmt(task: str, s: dict) -> list[str]:
    n = sum(s["slots"].values())
    idx = range(s["window"])
    tags = [_KIND_TAG.get(s["kind_of"].get(i, "pool"), "?") for i in idx]
    won = "  ".join(f"b{i}{t}:{s['ep_wins'].get(i, 0):>3}" for i, t in zip(idx, tags))
    mean = "  ".join(f"b{i}{t}:{100 * s['slot_mean'].get(i, 0.0):5.1f}%" for i, t in zip(idx, tags))
    kinds = "  ".join(
        f"{k}:{100 * v / n:.1f}%" for k, v in sorted(s["kinds"].items(), key=lambda kv: -kv[1])
    )
    return [
        f"[{task}] calls={s['calls']} episodes={s['episodes']} "
        f"mass={s['mass']:.4f} src={s['source']}   (P=pooled R=recent C=current)",
        f"    won    {won}   (episodes, {s['n_ep']} total)",
        f"    mean   {mean}",
        f"    winner {kinds}",
        (
            f"    age    mean {s['age_mean']:.1f}  p50 {s['age_p50']:.1f}  "
            f"max {s['age_max']} (policy calls back, history winners only)"
            if s["age_n"]
            else "    age    -- (the live observation won every call)"
        ),
        f"    padding winners {100 * s['padding']:.1f}%",
    ]


def _episodes(task_dir: Path) -> dict[str, list[dict]]:
    """{episode -> its calls}, one entry per `<task>/mem_attn/<session_id>.jsonl`."""
    mem = task_dir / "mem_attn"
    if not mem.is_dir():
        return {}
    out = {}
    for p in sorted(mem.glob("*.jsonl")):
        rows = _rows(p)
        if rows:
            out[p.stem] = sorted(rows, key=lambda r: r.get("call_index", 0))
    return out


def _episode_table(per_ep: dict[str, list[dict]], task_s: dict) -> list[str]:
    """One row per episode: its own mean over its own calls.

    Episodes differ in length by more than an order of magnitude (an early success ends
    in a handful of calls, a timeout runs to MAX_EP_STEPS), and the task line pools calls
    flat -- so it is dominated by the long ones. This table is the per-episode view that
    pooling hides, plus a macro mean that weights every episode equally.
    """
    idx = range(task_s["window"])
    tags = [_KIND_TAG.get(task_s["kind_of"].get(i, "pool"), "?") for i in idx]
    head = "  ".join(f"{f'b{i}{t}':>6}" for i, t in zip(idx, tags))
    lines = [
        f"    {'episode':<34} {'calls':>6} {'mass':>7}  {head}  "
        f"{'top-kind':<13} {'age':>5}"
    ]

    macro: dict[int, list[float]] = {i: [] for i in idx}
    for ep, rows in per_ep.items():
        s = summarize(rows)
        if not s:
            continue
        means = "  ".join(f"{100 * s['slot_mean'].get(i, 0.0):5.1f}%" for i in idx)
        for i in idx:
            macro[i].append(s["slot_mean"].get(i, 0.0))
        kind, cnt = s["kinds"].most_common(1)[0]
        age = f"{s['age_mean']:.0f}" if s["age_n"] else "--"
        share = f"{kind[:7]} {100 * cnt / sum(s['kinds'].values()):.0f}%"
        lines.append(
            f"    {ep[:34]:<34} {s['calls']:>6} {s['mass']:>7.4f}  {means}  "
            f"{share:<13} {age:>5}"
        )
    if macro[0]:
        avg = "  ".join(f"{100 * statistics.fmean(macro[i]):5.1f}%" for i in idx)
        lines.append(f"    {'mean over episodes (macro)':<34} {'':>6} {'':>7}  {avg}")
    return lines


def _task_report(task: str, per_ep: dict[str, list[dict]]) -> str:
    rows = [r for v in per_ep.values() for r in v]
    s = summarize(rows)
    if not s:
        return f"[{task}] no scoreable calls\n"
    lines = [f"Memory-attention attribution: {task}", ""]
    lines += _fmt(task, s)
    lines += ["", f"    per-episode ({len(per_ep)} episodes)"]
    lines += _episode_table(per_ep, s)
    return "\n".join(lines) + "\n"


def main(target: str) -> None:
    """`target` is a run dir (all tasks) or a single `<run>/<TASK>` dir."""
    root = Path(target)
    if (root / "mem_attn").is_dir():  # pointed at one task
        tasks = {root.name: root}
    else:
        tasks = {t: root / t for t in TASKS if (root / t / "mem_attn").is_dir()}
    if not tasks:
        print(f"[!] no mem_attn/*.jsonl under {root} — was GR00T_MEM_ATTR=1 set?")
        sys.exit(1)

    found = {}
    for task, task_dir in tasks.items():
        per_ep = _episodes(task_dir)
        if not per_ep:
            continue
        found[task] = per_ep
        report = _task_report(task, per_ep)
        out = task_dir / "mem_attention_report.txt"
        out.write_text(report)
        print(report)
        print(f"[i] wrote {out}\n")

    if len(found) > 1:  # run-level rollup across the tasks that have logs
        pooled = [r for per_ep in found.values() for v in per_ep.values() for r in v]
        s = summarize(pooled)
        if s:
            lines = [f"Memory-attention attribution: {root}", ""]
            for task, per_ep in found.items():
                ts = summarize([r for v in per_ep.values() for r in v])
                if ts:
                    lines += _fmt(task, ts) + [""]
            lines += _fmt(f"ALL ({len(found)} tasks)", s)
            summary = "\n".join(lines)
            print(summary)
            out = root / "mem_attention_summary.txt"
            out.write_text(summary + "\n")
            print(f"\n[i] wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])

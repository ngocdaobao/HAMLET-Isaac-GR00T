#!/usr/bin/env python3
"""Summarize the memory-attention attribution written by GR00T_MEM_ATTR=1.

Reads `<run_dir>/<TASK>/mem_attn/memory_attention.jsonl` (one JSON line per policy call,
written by Gr00tPolicy via gr00t/model/modules/memory_attribution.py) and reports, per
task, WHICH pooled step the action head read:

    mass        mean probability the action tokens put on the memory tokens at all.
                Near zero means memory is barely being used (or the key-moment gate
                zeroed it), and the rest of the row says little.
    slot        how often each pool slot won, oldest slot first. Flat means the pool is
                being read broadly; all mass on the last slot means the action head only
                ever reads the most recent block, i.e. the pool is doing no work a short
                FIFO window could not.
    age         how far back the winning step was captured, in policy calls. This is the
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
    """Highest-scoring block that is not the live observation."""
    pool = [s for s in row["steps"] if not s["is_current"]]
    return max(pool, key=lambda s: s["score"]) if pool else None


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
    ages = [a for a in (_age(r, w) for r, w in wins) if a is not None]
    return {
        "calls": len(rows),
        "episodes": len({r.get("session_id") or r.get("episode_id") for r in rows}),
        "mass": statistics.fmean(r["memory_attention_mass"] for r in rows),
        "slots": Counter(w["block_index"] for _, w in wins),
        "window": window,
        "age_mean": statistics.fmean(ages) if ages else float("nan"),
        "age_p50": statistics.median(ages) if ages else float("nan"),
        "age_max": max(ages) if ages else float("nan"),
        "padding": statistics.fmean(float(w["is_padding"]) for _, w in wins),
        "source": rows[0].get("source", "?"),
    }


def _fmt(task: str, s: dict) -> list[str]:
    n = sum(s["slots"].values())
    slots = "  ".join(
        f"b{i}:{100 * s['slots'].get(i, 0) / n:4.1f}%" for i in range(s["window"] - 1)
    )
    return [
        f"[{task}] calls={s['calls']} episodes={s['episodes']} "
        f"mass={s['mass']:.4f} src={s['source']}",
        f"    slot   {slots}",
        f"    age    mean {s['age_mean']:.1f}  p50 {s['age_p50']:.1f}  max {s['age_max']}"
        " (policy calls back)",
        f"    padding winners {100 * s['padding']:.1f}%",
    ]


def main(run_dir: str) -> None:
    run = Path(run_dir)
    single = run / "mem_attn" / "memory_attention.jsonl"
    if single.is_file():  # pointed at one task directory
        found = {run.name: _rows(single)}
    else:
        found = {}
        for t in TASKS:
            p = run / t / "mem_attn" / "memory_attention.jsonl"
            if p.is_file():
                found[t] = _rows(p)
    if not found:
        print(f"[!] no memory_attention.jsonl under {run} — was GR00T_MEM_ATTR=1 set?")
        sys.exit(1)

    lines = [f"Memory-attention attribution: {run}", ""]
    for task, rows in found.items():
        s = summarize(rows)
        lines += _fmt(task, s) if s else [f"[{task}] no scoreable calls"]
        lines.append("")
    pooled = [r for rows in found.values() for r in rows]
    s = summarize(pooled)
    if s:
        lines += _fmt(f"ALL ({len(found)} tasks)", s)

    summary = "\n".join(lines)
    print(summary)
    out = run / "mem_attention_summary.txt"
    out.write_text(summary + "\n")
    print(f"\n[i] wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])

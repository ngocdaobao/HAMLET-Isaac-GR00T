# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which pooled step is the action head actually reading?

At inference the memory pool holds K blocks of n_q tokens (`Gr00tN1d6ActionHead.
process_mem_cache` -> `mem_seq`), but the action head never sees them: the memory
transformer collapses the whole window into the CURRENT block's n_q tokens, and only
those reach the DiT. So "attention on a pooled step" is a two-hop quantity and has to be
composed:

    hop 1 (DiT cross-attention)      action tokens -> the n_q memory tokens
    hop 2 (memory transformer)       current block's n_q tokens -> each of the K blocks

Hop 2 is composed across the memory transformer's layers by attention rollout
(Abnar & Zuidema 2020): each layer's head-averaged attention is mixed with the identity
to account for the residual stream, row-renormalized, and the layers are multiplied.
Hop 1 weights the n_q rows of that rollout by how much probability mass the action
queries put on each memory token, summed over every cross-attention block and every
denoising step. The product, summed over the n_q tokens of a block, is the score of that
block; blocks are then labelled with the pool step ids they came from.

Both hops are read with hooks, so nothing in the training path changes and the probe can
be attached to an already-loaded policy:

    from gr00t.model.modules.memory_attribution import attach_memory_probe

    with attach_memory_probe(policy) as probe:
        for step in rollout:
            action = policy.get_action(obs, options)
            rep = probe.report()[0]
            print(rep.summary())        # top pooled steps for this call
            probe.dump_jsonl("mem_attr.jsonl")

`mem_cond_type="adaln"` has no hop 1 (memory is mean-pooled into the DiT timestep
embedding instead of cross-attended), so the n_q weights are uniform there -- which is
exactly what mean pooling implies -- and `source` on the report says so.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import math
import os
from typing import Any, Iterator

import torch

from gr00t.model.modules.memory import MemoryTransformer, _apply_rope


@dataclass
class MemoryStepAttribution:
    """One block of `mem_seq`, i.e. one step the pool is holding."""

    block_index: int  # position in mem_seq, 0 = oldest
    step_id: int | None  # episode step the block was captured at (None = unknown)
    score: float  # share of the action head's memory attention, sums to 1 over blocks
    is_current: bool  # the live observation (always the last block)
    is_padding: bool  # warm-up duplicate: the pool had no distinct block for this slot

    def __repr__(self) -> str:  # compact, these get printed in rollout loops
        tag = "*" if self.is_current else ("~" if self.is_padding else " ")
        return f"{tag}b{self.block_index}(step={self.step_id}):{self.score:.3f}"


@dataclass
class MemoryAttributionReport:
    """Per batch row result of one policy call."""

    episode_id: Any
    query_step_id: int | None
    steps: list[MemoryStepAttribution]
    token_weights: list[float]  # hop-1 weight of each of the n_q memory tokens
    memory_attention_mass: float  # mean prob. mass action queries put on memory tokens
    source: str  # "cross_attn" or "adaln_uniform"
    memory_mode: str  # "zoo" or "window"
    n_q: int
    window: int

    def top(self, k: int = 1, exclude_current: bool = True) -> list[MemoryStepAttribution]:
        """The k highest-scoring blocks, best first.

        The live observation is excluded by default: it is in `mem_seq` but it is not a
        pooled memory, and it usually dominates (the memory transformer's own block is
        the one place every current token can attend without crossing a block).
        """
        pool = [s for s in self.steps if not (exclude_current and s.is_current)]
        return sorted(pool, key=lambda s: s.score, reverse=True)[:k]

    @property
    def best(self) -> MemoryStepAttribution | None:
        top = self.top(1)
        return top[0] if top else None

    def summary(self, k: int = 3) -> str:
        top = ", ".join(str(s) for s in self.top(k))
        return (
            f"[mem-attr] ep={self.episode_id} step={self.query_step_id} "
            f"src={self.source} mass={self.memory_attention_mass:.4f} top{k}=[{top}]"
        )

    def to_dict(self) -> dict:
        return asdict(self)


class MemoryAttentionProbe:
    """Reads the two attention hops above off a live `Gr00tN1d6ActionHead`.

    Args:
        action_head:     the action head owning `memory_transformer` and the DiT.
        residual_alpha:  weight of the identity in the rollout mix
                         `alpha*I + (1-alpha)*A`. 0.5 is the standard rollout; 0 ignores
                         the residual stream and makes the deepest layer dominate.
        query_slice:     "action" scores only the action tokens (the ones that become
                         the trajectory), "all" also counts the state token.
        denoise_reduce:  "mean" averages hop 1 over the denoising steps, "last" keeps
                         only the final one (closest to the emitted action).
    """

    def __init__(
        self,
        action_head,
        *,
        residual_alpha: float = 0.5,
        query_slice: str = "action",
        denoise_reduce: str = "mean",
    ):
        if getattr(action_head, "memory_transformer", None) is None:
            raise ValueError(
                "MemoryAttentionProbe needs an action head with a memory transformer "
                "(hamlet_mode='finetune' and memory_num_layers > 0)."
            )
        if query_slice not in ("action", "all"):
            raise ValueError(f"query_slice must be 'action' or 'all', got {query_slice!r}")
        if denoise_reduce not in ("mean", "last"):
            raise ValueError(f"denoise_reduce must be 'mean' or 'last', got {denoise_reduce!r}")

        self.action_head = action_head
        self.memory_transformer: MemoryTransformer = action_head.memory_transformer
        self.residual_alpha = residual_alpha
        self.query_slice = query_slice
        self.denoise_reduce = denoise_reduce

        self.n_q = int(getattr(action_head, "_mem_tokens_per_step"))
        self.window = int(self.memory_transformer.T)
        self.mem_cond_type = getattr(action_head, "mem_cond_type", "cross_attn")
        self.memory_mode = getattr(action_head, "memory_mode", "window")

        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        # Per-call buffers, cleared when the memory transformer starts a new forward.
        self._mem_layers: list[torch.Tensor] = []  # per layer, (B, L, L) head-averaged
        self._tok_w: torch.Tensor | None = None  # (B, n_q) unnormalized hop-1 weights
        self._mass: torch.Tensor | None = None  # (B,) prob. mass on memory per xattn call
        self._n_xattn = 0
        self._pending: list[dict] = []  # reports queued for dump_jsonl

    # ---------------------------------------------------------------- attach / detach

    def attach(self) -> "MemoryAttentionProbe":
        if self._handles:
            return self
        mt = self.memory_transformer
        self._handles.append(mt.register_forward_pre_hook(self._on_memory_forward))
        for block in mt.blocks:
            self._handles.append(
                block.attn.register_forward_pre_hook(self._on_memory_attn, with_kwargs=True)
            )
        for block in self._dit_blocks():
            self._handles.append(
                block.attn1.register_forward_pre_hook(self._on_dit_attn, with_kwargs=True)
            )
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _dit_blocks(self):
        model = getattr(self.action_head, "model", None)
        blocks = getattr(model, "transformer_blocks", None)
        if blocks is None:
            raise ValueError("action head has no DiT `transformer_blocks` to hook")
        return blocks

    def __enter__(self) -> "MemoryAttentionProbe":
        return self.attach()

    def __exit__(self, *exc) -> None:
        self.detach()

    def reset(self) -> None:
        self._mem_layers = []
        self._tok_w = None
        self._mass = None
        self._n_xattn = 0

    # ------------------------------------------------------------------------- hooks

    @torch.no_grad()
    def _on_memory_forward(self, module, args):
        # A new memory-transformer forward = a new policy call: the DiT weights collected
        # so far belong to the previous call's denoising loop.
        self.reset()

    @torch.no_grad()
    def _on_memory_attn(self, module, args, kwargs):
        """Recompute one memory-transformer layer's attention.

        `_Attention.forward(x, attn_mask, cos, sin)` runs SDPA, which returns no weights,
        so they are recomputed here from the same projections and the same RoPE tables.
        Cheap: L = window * n_q.
        """
        x = kwargs.get("x", args[0] if args else None)
        attn_mask = kwargs.get("attn_mask", args[1] if len(args) > 1 else None)
        cos = kwargs.get("cos", args[2] if len(args) > 2 else None)
        sin = kwargs.get("sin", args[3] if len(args) > 3 else None)
        if x is None or cos is None or sin is None:
            return

        B, L, _ = x.shape
        h, d = module.num_heads, module.head_dim
        q = module.q_proj(x).view(B, L, h, d).transpose(1, 2)
        k = module.k_proj(x).view(B, L, h, d).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)
        logits = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(d)
        if attn_mask is not None:
            logits = logits + attn_mask.float()
        probs = torch.softmax(logits, dim=-1).mean(dim=1)  # (B, L, L), head-averaged
        self._mem_layers.append(probs.detach().cpu())

    @torch.no_grad()
    def _on_dit_attn(self, module, args, kwargs):
        """Hop 1: how much of each action query's cross-attention lands on the memory
        tokens, which are the LAST n_q columns of the DiT's key/value sequence.

        Self-attention blocks (no `encoder_hidden_states`) carry no memory columns and
        are skipped. Blocks whose mask hides the memory tokens -- the image-only
        cross-attention blocks of `AlternateVLDiT` -- contribute (near) zero mass, which
        is the correct weight for them, so no block needs special casing.
        """
        hidden_states = kwargs.get("hidden_states", args[0] if args else None)
        enc = kwargs.get("encoder_hidden_states", args[1] if len(args) > 1 else None)
        mask = kwargs.get("attention_mask", args[2] if len(args) > 2 else None)
        if hidden_states is None or enc is None:
            return  # self-attention block
        if self.mem_cond_type == "adaln":
            return  # memory never enters the KV; weights stay uniform
        if enc.shape[1] < self.n_q:
            return

        q_rows = hidden_states
        if self.query_slice == "action":
            horizon = int(getattr(self.action_head, "action_horizon", 0))
            if 0 < horizon < hidden_states.shape[1]:
                q_rows = hidden_states[:, -horizon:, :]

        if getattr(module, "norm_cross", None) is not None:
            enc = module.norm_encoder_hidden_states(enc)
        q = module.to_q(q_rows)
        k = module.to_k(enc)
        B, Lq, _ = q.shape
        h = module.heads
        d = q.shape[-1] // h
        q = q.view(B, Lq, h, d).transpose(1, 2)
        k = k.view(B, k.shape[1], h, -1).transpose(1, 2)
        if getattr(module, "norm_q", None) is not None:
            q = module.norm_q(q)
        if getattr(module, "norm_k", None) is not None:
            k = module.norm_k(k)

        logits = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(d)
        if mask is not None:
            m = mask
            while m.dim() < 4:
                m = m.unsqueeze(1)
            if m.dtype == torch.bool:
                logits = logits.masked_fill(~m, float("-inf"))
            else:
                logits = logits + m.float()
        probs = torch.nan_to_num(torch.softmax(logits, dim=-1))  # all-masked rows -> 0
        mem = probs[..., -self.n_q :].mean(dim=1).mean(dim=1)  # (B, n_q), head+query mean

        if self.denoise_reduce == "last" and self._n_xattn and self._is_new_denoise_step():
            self._tok_w = None
            self._mass = None
            self._n_xattn = 0
        mass = mem.sum(dim=-1).detach().cpu()
        mem = mem.detach().cpu()
        self._tok_w = mem if self._tok_w is None else self._tok_w + mem
        self._mass = mass if self._mass is None else self._mass + mass
        self._n_xattn += 1

    def _is_new_denoise_step(self) -> bool:
        """`denoise_reduce="last"` keeps only the final denoising pass. A pass is a full
        sweep of the DiT's cross-attention blocks, so a new one starts once as many have
        been seen as the DiT has."""
        n_cross = sum(1 for b in self._dit_blocks() if b.cross_attention_dim is not None)
        return n_cross > 0 and self._n_xattn >= n_cross

    # ------------------------------------------------------------------------ report

    @torch.no_grad()
    def rollout(self) -> torch.Tensor | None:
        """Hop 2: (B, n_q, window) mass each current-block token puts on each block."""
        if not self._mem_layers:
            return None
        a = self.residual_alpha
        r = None
        for probs in self._mem_layers:
            L = probs.shape[-1]
            eye = torch.eye(L, dtype=probs.dtype).unsqueeze(0)
            mixed = a * eye + (1.0 - a) * probs
            mixed = mixed / mixed.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            r = mixed if r is None else torch.bmm(mixed, r)
        B, L, _ = r.shape
        n_q, T = self.n_q, self.window
        if L != T * n_q:  # a differently-shaped memory transformer: bail out loudly
            raise RuntimeError(f"memory attention is {L}x{L}, expected {T * n_q}x{T * n_q}")
        cur = r[:, -n_q:, :].view(B, n_q, T, n_q).sum(dim=-1)  # (B, n_q, T)
        return cur

    @torch.no_grad()
    def report(self) -> list[MemoryAttributionReport]:
        """Attribution for the most recent policy call, one entry per batch row.

        Returns an empty list when the probe saw no memory-transformer forward since the
        last call (e.g. the policy ran without HAMLET memory).
        """
        cur = self.rollout()
        if cur is None:
            return []
        B, n_q, T = cur.shape

        if self._tok_w is None:
            # adaln (mean pooling -> every memory token counts the same), or a call whose
            # cross-attention never exposed the memory columns.
            w = torch.full((B, n_q), 1.0 / n_q)
            mass = torch.zeros(B)
            source = "adaln_uniform" if self.mem_cond_type == "adaln" else "uniform_fallback"
        else:
            w = self._tok_w
            total = w.sum(dim=-1, keepdim=True)
            # A row with no mass anywhere (memory masked out by the key-moment gate)
            # would divide by zero; fall back to uniform so the block ranking still
            # reflects hop 2 instead of becoming NaN.
            w = torch.where(total > 1e-12, w / total.clamp_min(1e-12), torch.full_like(w, 1.0 / n_q))
            mass = self._mass / max(self._n_xattn, 1)
            source = "cross_attn"

        scores = torch.einsum("bq,bqt->bt", w.to(cur.dtype), cur)
        scores = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        step_ids = self._step_ids(B)
        episode_ids = getattr(self.action_head, "_last_mem_episode_ids", None) or [None] * B

        reports = []
        for b in range(B):
            ids = step_ids[b]
            seen: set = set()
            blocks = []
            for i, sid in enumerate(ids):
                is_current = i == T - 1
                # `process_mem_cache` left-pads a short pool by repeating its oldest
                # block, so a repeated id is a padding slot, not a real memory.
                is_pad = (not is_current) and sid is not None and sid in seen
                seen.add(sid)
                blocks.append(
                    MemoryStepAttribution(
                        block_index=i,
                        step_id=sid,
                        score=float(scores[b, i]),
                        is_current=is_current,
                        is_padding=is_pad,
                    )
                )
            reports.append(
                MemoryAttributionReport(
                    episode_id=episode_ids[b] if b < len(episode_ids) else None,
                    query_step_id=ids[-1],
                    steps=blocks,
                    token_weights=[float(x) for x in w[b]],
                    memory_attention_mass=float(mass[b]),
                    source=source,
                    memory_mode=self.memory_mode,
                    n_q=n_q,
                    window=T,
                )
            )
        self._pending = [r.to_dict() for r in reports]
        return reports

    def _step_ids(self, B: int) -> list[list[int | None]]:
        """Episode step id of each block of `mem_seq`, oldest first.

        Only the zoo pool knows them (it records the ids it stacked); the rolling FIFO
        window is contiguous, so its blocks are labelled with relative offsets
        -(T-1)..0 instead.
        """
        T = self.window
        if self.memory_mode == "zoo":
            ids = getattr(self.action_head, "_last_mem_step_ids", None)
            if ids and len(ids) == B:
                return [list(row) for row in ids]
        return [list(range(-(T - 1), 1)) for _ in range(B)]

    # -------------------------------------------------------------------- convenience

    def dump_jsonl(self, path: str, extra: dict | list[dict] | None = None) -> None:
        """Append the last `report()` (one JSON object per batch row) to `path`.

        `extra` merges caller fields into the rows -- a dict for all of them, or a list
        aligned with the batch (e.g. the session id and call index of each row).
        """
        if not self._pending:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "a") as f:
            for i, row in enumerate(self._pending):
                if isinstance(extra, dict):
                    row = {**row, **extra}
                elif extra is not None and i < len(extra):
                    row = {**row, **extra[i]}
                f.write(json.dumps(row, default=str) + "\n")


def _resolve_action_head(target):
    """Accept a Gr00tPolicy, a Gr00tN1d6, or the action head itself."""
    for obj in (target, getattr(target, "model", None)):
        if obj is None:
            continue
        if getattr(obj, "memory_transformer", None) is not None:
            return obj
        head = getattr(obj, "action_head", None)
        if head is not None:
            return head
        inner = getattr(obj, "model", None)  # Gr00tPolicy.model -> Gr00tN1d6
        if inner is not None and getattr(inner, "action_head", None) is not None:
            return inner.action_head
    raise ValueError(f"could not find a Gr00tN1d6ActionHead on {type(target).__name__}")


@contextmanager
def attach_memory_probe(target, **kwargs) -> Iterator[MemoryAttentionProbe]:
    """Attach a probe to a policy/model/action head for the duration of the block."""
    probe = MemoryAttentionProbe(_resolve_action_head(target), **kwargs)
    probe.attach()
    try:
        yield probe
    finally:
        probe.detach()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HAMLET time-contrastive learning (TCL) head for the N1.6 Eagle backbone.

  z = L2-normalize( moment_to_repr( mean-pool over n_q moment tokens ) )
  L = CE( [sim(z_a, z_p), sim(z_a, z_n)] / tau, labels=0 )

Trainable: backbone.moment_tokens + self.moment_to_repr. The action expert is not
used (replaced by this head when hamlet_mode == "tcl").
"""

from __future__ import annotations

import logging

import torch
from torch import nn
from torch.nn import functional as F
from transformers.feature_extraction_utils import BatchFeature


logger = logging.getLogger(__name__)


class Gr00tN1d6TCLHead(nn.Module):
    """Time-Contrastive Learning head."""

    supports_gradient_checkpointing = False

    def __init__(self, backbone_embedding_dim: int, tcl_tau: float = 0.07):
        super().__init__()
        d = backbone_embedding_dim
        # 2-layer Linear(d->d) + SiLU + Linear(d->d); L2 normalization is applied at
        # forward via F.normalize.
        self.moment_to_repr = nn.Sequential(
            nn.Linear(d, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
        self.reset_parameters()
        self.tcl_tau = tcl_tau
        self.mask_token = None  # stub for trainer compatibility
        self._debug_dumped = False

    def reset_parameters(self):
        """Initialize the projection MLP. Safe to call after `from_pretrained`, which
        leaves these as missing keys materialized from `torch.empty` (usually all-zero
        pages -> z == 0 -> constant loss = ln 2 and exactly zero gradients)."""
        for m in self.moment_to_repr.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def set_trainable_parameters(self, *_args, **_kwargs):
        for p in self.parameters():
            p.requires_grad = True

    def set_frozen_modules_to_eval_mode(self):
        pass

    def _moment_repr(self, backbone_output: BatchFeature) -> tuple[torch.Tensor, dict]:
        """Mean-pool the n_q moment-token tail, project, L2-normalize -> (B, d).

        Also returns the pre-normalization magnitudes: a dead head (all-zero projection
        weights, or a backbone tail that never varies) is invisible in the loss but shows
        up immediately as ``repr_norm == 0``.
        """
        feats = backbone_output["backbone_features"]  # (B, T, d)
        n_q = int(backbone_output["n_moment_tokens"])
        mq = feats[:, -n_q:, :]
        pooled = mq.mean(dim=1)
        # Cast to the projection MLP's dtype for autocast cleanliness.
        proj_dtype = next(self.moment_to_repr.parameters()).dtype
        h = self.moment_to_repr(pooled.to(proj_dtype))
        z = F.normalize(h, dim=-1, eps=1e-8)
        stats = {
            "pooled_norm": pooled.detach().float().norm(dim=-1).mean(),
            "repr_norm": h.detach().float().norm(dim=-1).mean(),
        }
        return z, stats

    def forward(
        self,
        anchor_output: BatchFeature,
        aug_output: BatchFeature,
        neg_output: BatchFeature,
        action_input: BatchFeature,
    ) -> dict:
        z_a, st_a = self._moment_repr(anchor_output)
        z_p, st_p = self._moment_repr(aug_output)
        z_n, st_n = self._moment_repr(neg_output)

        if not self._debug_dumped:
            self._debug_dumped = True
            self._dump_debug(anchor_output, aug_output, neg_output, (st_a, st_p, st_n))

        sim_ap = torch.sum(z_a * z_p, dim=-1, keepdim=True)  # (B, 1)
        sim_an = torch.sum(z_a * z_n, dim=-1, keepdim=True)  # (B, 1)
        logits = torch.cat([sim_ap, sim_an], dim=1) / self.tcl_tau  # (B, 2)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            tcl_pos_sim = sim_ap.mean()
            tcl_neg_sim = sim_an.mean()
            tcl_repr_norm = torch.stack([st_a["repr_norm"], st_p["repr_norm"], st_n["repr_norm"]]).mean()

        return {
            "loss": loss,
            "tcl_pos_sim": tcl_pos_sim.detach(),
            "tcl_neg_sim": tcl_neg_sim.detach(),
            "tcl_repr_norm": tcl_repr_norm.detach(),
        }

    @torch.no_grad()
    def _dump_debug(self, anchor_output, aug_output, neg_output, stats) -> None:
        """One-shot dump on the first forward: tells apart the three ways this stage dies
        (dead projection, identical anchor/aug/neg streams, dead moment-token tail)."""
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
                return
        except Exception:
            pass

        lines = ["[TCL-DEBUG] first forward:"]
        for i, m in enumerate(self.moment_to_repr):
            if isinstance(m, nn.Linear):
                lines.append(
                    f"  moment_to_repr[{i}]: |W|={m.weight.detach().float().abs().max().item():.3e} "
                    f"|b|={m.bias.detach().float().abs().max().item():.3e}"
                )
        for name, out, st in zip(("anchor", "aug", "neg"), (anchor_output, aug_output, neg_output), stats):
            feats = out["backbone_features"]
            n_q = int(out["n_moment_tokens"])
            tail = feats[:, -n_q:, :].detach().float()
            lines.append(
                f"  {name}: feats={tuple(feats.shape)} n_q={n_q} "
                f"tail|mu|={tail.abs().mean().item():.3e} tail_std={tail.std().item():.3e} "
                f"pooled_norm={st['pooled_norm'].item():.3e} repr_norm={st['repr_norm'].item():.3e}"
            )
        a = anchor_output["backbone_features"].detach().float()
        for name, out in (("aug", aug_output), ("neg", neg_output)):
            b = out["backbone_features"].detach().float()
            same = a.shape == b.shape and torch.equal(a, b)
            lines.append(f"  anchor vs {name} backbone_features identical: {same}")
        logger.warning("\n".join(lines))

    @torch.no_grad()
    def get_action(self, *args, **kwargs):
        raise RuntimeError("TCL head does not support get_action; use a Stage-2 checkpoint.")

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

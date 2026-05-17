"""Label-as-query multi-label decoder over per-tile DINOv2 CLS embeddings.

Implements the architecture in
``user/murilo/experiments/003-tileq-decoder/spec.md`` §2.1: ~4 M-param head
sitting on top of a frozen ViT-B DINOv2 backbone (the backbone is NOT part of
this module). For each input quadrat:

    tile_cls : (B, T=16, d_in=768)   raw, non-L2-normalised CLS tokens
    --> proj_kv : (B, T, d_model=384)
    --> cross-attention from G=64 group queries
    --> norm + FFN + norm
    --> group-FFN classifier head: (B, G, classes_per_group)
    --> reshape (B, G * classes_per_group) and trim to n_classes
    logits : (B, n_classes=7806)

Group-FFN keeps the query count tractable for 7806 classes while still updating
all 64 group queries every batch (vs gradient starvation if we used 7806
individual queries on a long-tail label space). Reference: ML-Decoder
(Ridnik et al., WACV 2023).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TileQDecoder(nn.Module):
    def __init__(
        self,
        n_classes: int = 7806,
        n_groups: int = 64,
        d_in: int = 768,
        d_model: int = 384,
        n_heads: int = 8,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_groups = n_groups
        self.d_in = d_in
        self.d_model = d_model
        self.classes_per_group = math.ceil(n_classes / n_groups)

        self.proj_kv = nn.Linear(d_in, d_model)
        self.queries = nn.Parameter(torch.randn(n_groups, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.group_heads = nn.Parameter(
            torch.randn(n_groups, d_model, self.classes_per_group) * 0.02
        )
        self.group_bias = nn.Parameter(torch.zeros(n_groups, self.classes_per_group))

    def forward(self, tile_cls: torch.Tensor) -> torch.Tensor:
        # tile_cls: (B, T, d_in)
        B = tile_cls.size(0)
        kv = self.proj_kv(tile_cls)                              # (B, T, D)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)          # (B, G, D)
        attn_out, _ = self.cross_attn(q, kv, kv)                 # (B, G, D)
        h = self.norm1(q + attn_out)
        h = self.norm2(h + self.ffn(h))                          # (B, G, D)
        # group-FFN: (B, G, D) · (G, D, C') -> (B, G, C')
        logits = torch.einsum("bgd,gdc->bgc", h, self.group_heads) + self.group_bias
        return logits.reshape(B, -1)[:, : self.n_classes]        # (B, n_classes)

"""Minimal LoRA implementation for timm ViT attention layers.

Wraps nn.Linear modules with a low-rank delta:

    y = base(x) + dropout(x) @ A.T @ B.T * (alpha / r)

The base Linear is kept and frozen; only A, B are trained.

Usage:
    inject_lora_into_attention(model, r=8, alpha=16, dropout=0.0)
    # Trainable params now include only the A/B matrices added to attn.qkv
    # and (optionally) attn.proj across all blocks.

Ported verbatim from Exp-9B_Improved_KNN_Retrieval/lora.py — works for any
ViT (ViT-B 12 blocks or ViT-L 24 blocks) since module discovery is by
attribute name (.attn.qkv), not block count.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """A frozen nn.Linear augmented by a trainable low-rank delta."""

    def __init__(
        self,
        base: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        in_f, out_f = base.in_features, base.out_features
        self.r = r
        self.scaling = alpha / r

        self.lora_A = nn.Parameter(torch.zeros(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return base_out + lora_out * self.scaling


def _replace_linear(parent: nn.Module, attr_name: str, r: int, alpha: float, dropout: float) -> bool:
    child = getattr(parent, attr_name, None)
    if not isinstance(child, nn.Linear):
        return False
    setattr(parent, attr_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
    return True


def inject_lora_into_attention(
    model: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    include_proj: bool = False,
) -> List[str]:
    """Inject LoRA into every attn.qkv (and optionally attn.proj) in a ViT.

    Works on any timm ViT (ViT-B has 12 blocks, ViT-L has 24); discovery is
    by attribute name, not by block count.
    """
    touched = []
    for name, module in model.named_modules():
        if not hasattr(module, "attn"):
            continue
        attn = module.attn
        if _replace_linear(attn, "qkv", r=r, alpha=alpha, dropout=dropout):
            touched.append(f"{name}.attn.qkv")
        if include_proj and _replace_linear(attn, "proj", r=r, alpha=alpha, dropout=dropout):
            touched.append(f"{name}.attn.proj")
    return touched


def freeze_non_lora(model: nn.Module) -> Tuple[int, int]:
    n_trainable = 0
    n_frozen = 0
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad = True
            n_trainable += p.numel()
        else:
            p.requires_grad = False
            n_frozen += p.numel()
    return n_trainable, n_frozen


def lora_state_dict(model: nn.Module) -> dict:
    return {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }


def load_lora_state(model: nn.Module, state: dict) -> None:
    own = model.state_dict()
    missing = [k for k in state if k not in own]
    if missing:
        raise KeyError(f"LoRA keys not present in model: {missing[:5]}...")
    for k, v in state.items():
        own[k].copy_(v)

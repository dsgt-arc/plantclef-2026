"""Asymmetric Loss for multi-label classification.

Implementation of ASL from Ridnik et al. ICCV 2021 (Alibaba-MIIL/ASL),
matching the formulation in ``user/murilo/experiments/003-tileq-decoder/spec.md``
§2.2:

    p     = sigmoid(logit)
    p_-   = max(p - margin, 0)              # equivalently  1 - p_- = clamp((1-p) + m, 0, 1)
    L_+   = (1 - p)^gamma_pos · log(p)      # for positives (y = 1)
    L_-   = (p_-)^gamma_neg  · log(1 - p_-) # for negatives (y = 0)
    loss  = -(y · L_+ + (1 - y) · L_-)

Spec defaults: gamma_neg=4, gamma_pos=0, margin=0.05. With those values,
positives use plain BCE while negatives are zero-loss when p < margin and
focal-weighted (upweight hard / downweight easy) otherwise.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        margin: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean_per_sample",
    ):
        super().__init__()
        if reduction not in {"none", "sum", "mean", "mean_per_sample"}:
            raise ValueError(f"unsupported reduction: {reduction!r}")
        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.margin = float(margin)
        self.eps = float(eps)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C) raw scores; targets: (B, C) in {0, 1}
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        # margin shift: equivalent to p_- = max(p - m, 0); when p < m, xs_neg -> 1
        if self.margin > 0:
            xs_neg = (xs_neg + self.margin).clamp(max=1.0)

        log_pos = torch.log(xs_pos.clamp(min=self.eps))
        log_neg = torch.log(xs_neg.clamp(min=self.eps))
        loss = targets * log_pos + (1.0 - targets) * log_neg

        # asymmetric focal weight: (1 - p_t)^gamma where p_t is "correct-class prob"
        if self.gamma_pos > 0 or self.gamma_neg > 0:
            with torch.no_grad():
                pt = xs_pos * targets + xs_neg * (1.0 - targets)
                gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
                weight = (1.0 - pt).pow(gamma)
            loss = loss * weight

        loss = -loss

        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "mean":
            return loss.mean()
        # default "mean_per_sample": sum over classes, mean over batch
        return loss.sum(dim=-1).mean()

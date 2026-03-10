"""Phase 4 — Loss functions for student training.

Two pseudo-label loss variants (ablation) + multi-view consistency loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCENegSamplingLoss(nn.Module):
    """Variant A: BCE with explicit positive/negative sampling.

    Positives come from pseudo-labels (target=1). Negatives are randomly
    sampled from the complement (target=0). This avoids assuming all
    non-pseudo-label species are truly absent.
    """

    def __init__(self, num_classes: int = 7806, num_negatives: int = 64):
        super().__init__()
        self.num_classes = num_classes
        self.num_negatives = num_negatives

    def forward(
        self,
        logits: torch.Tensor,              # [B, C]
        pseudo_labels: list[torch.Tensor],  # list of [L_i] int tensors
    ) -> torch.Tensor:
        B, C = logits.shape
        device = logits.device

        target = torch.zeros(B, C, device=device)
        weight = torch.zeros(B, C, device=device)

        for i, labels in enumerate(pseudo_labels):
            pos = labels.to(device, dtype=torch.long)
            if len(pos) == 0:
                continue

            # Mark positives
            target[i, pos] = 1.0
            weight[i, pos] = 1.0

            # Sample negatives (excluding positives)
            neg_mask = torch.ones(C, dtype=torch.bool, device=device)
            neg_mask[pos] = False
            neg_pool = neg_mask.nonzero(as_tuple=True)[0]
            n_neg = min(self.num_negatives, len(neg_pool))
            perm = torch.randperm(len(neg_pool), device=device)[:n_neg]
            weight[i, neg_pool[perm]] = 1.0

        if weight.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = F.binary_cross_entropy_with_logits(
            logits, target, weight=weight, reduction="sum"
        )
        return loss / weight.sum()


class SoftDistillationLoss(nn.Module):
    """Variant B (recommended): KL-divergence via BCE with soft teacher targets.

    For independent Bernoulli outputs, KL(teacher || student) equals
    BCE(student_logits, teacher_probs) up to a constant (teacher entropy).
    Minimizing BCE with soft targets is equivalent to minimizing KL-divergence.

    This preserves the teacher's uncertainty across all 7806 classes —
    the student learns the full probability landscape, not just hard 0/1.
    """

    def forward(
        self,
        logits: torch.Tensor,         # [B, C]
        teacher_probs: torch.Tensor,  # [B, C] in [0, 1]
    ) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, teacher_probs, reduction="mean"
        )


class ConsistencyLoss(nn.Module):
    """Multi-view consistency loss in probability space.

    Forces the student to predict the same species regardless of which
    geometric view (orig / flip / rot90) of the tiles it receives.

    Flip and rot90 predictions are stop-gradiented (MeanTeacher style):
    only the original view's logits receive gradients from this loss.
    """

    def forward(
        self,
        logits_orig: torch.Tensor,   # [B, C]
        logits_flip: torch.Tensor,   # [B, C] — detached
        logits_rot90: torch.Tensor,  # [B, C] — detached
    ) -> torch.Tensor:
        p_orig = torch.sigmoid(logits_orig)
        p_flip = torch.sigmoid(logits_flip)
        p_rot90 = torch.sigmoid(logits_rot90)

        loss_flip = F.mse_loss(p_orig, p_flip)
        loss_rot90 = F.mse_loss(p_orig, p_rot90)

        return (loss_flip + loss_rot90) / 2.0
